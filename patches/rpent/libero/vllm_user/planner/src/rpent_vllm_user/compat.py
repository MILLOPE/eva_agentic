"""Compatibility boundary between the migrated planner and eva's RPent baseline.

The planner source was developed against a newer RPent checkout. This module
keeps correction-only observability helpers local to the planner snapshot while
using the current RPent Toolkit, ToolResult, and Dashboard sink as the owners.
It does not create another event bus or another tool registry.
"""

from __future__ import annotations

import traceback
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from rpent.dashboard.events import (
    DashboardEventSink,
    NullDashboardEventSink,
    TranscriptEvent,
    UsageEvent,
)
from rpent.planner.base import PlannerResult
from rpent.tools.toolkit import ToolResult

try:
    from rpent.dashboard.events import (
        CapabilityCatalogEntry,
        CapabilityCatalogEvent,
        PlannerRequestUsageEvent,
        PlannerSystemContextEvent,
        ToolCallEvent,
    )
except ImportError:

    @dataclass(frozen=True, slots=True)
    class CapabilityCatalogEntry:
        """Planner-local catalog record for newer RPent observers."""

        tool_name: str
        summary: str
        category: str = "generic"
        parameter_names: tuple[str, ...] = ()
        source: str = "toolkit"
        skill_id: str | None = None
        skill_version: str | None = None

    @dataclass(frozen=True, slots=True)
    class CapabilityCatalogEvent:
        """Planner-local catalog event for newer RPent observers."""

        entries: tuple[CapabilityCatalogEntry, ...]

    @dataclass(frozen=True, slots=True)
    class PlannerSystemContextEvent:
        """Planner-local system-context event for newer RPent observers."""

        text: str
        source: str = "provided"

    @dataclass(frozen=True, slots=True)
    class PlannerRequestUsageEvent:
        """Planner-local detailed usage event for newer RPent observers."""

        turn: int | None
        input_tokens: int
        output_tokens: int
        request_index: int | None = None
        usage_sample_index: int | None = None
        model: str | None = None
        temperature: float | None = None
        max_tokens: int | None = None
        enable_thinking: bool | None = None
        parallel_tool_calls: bool | None = None
        tool_choice: str | None = None
        message_count: int | None = None
        message_chars: int | None = None
        tool_count: int | None = None
        elapsed_s: float | None = None
        request_text_bytes: int | None = None
        request_image_bytes: int | None = None
        request_wire_bytes: int | None = None
        compacted: bool | None = None
        context_overflow_retry: bool | None = None
        cached_input_tokens: int | None = None
        reasoning_output_tokens: int | None = None
        system_message_chars: int | None = None
        user_message_chars: int | None = None
        assistant_message_chars: int | None = None
        tool_result_chars: int | None = None
        image_count: int | None = None
        tool_schema_bytes: int | None = None
        system_prompt: str | None = None
        context_previews: dict[str, Any] | None = None

    @dataclass(frozen=True, slots=True)
    class ToolCallEvent:
        """Planner-local lifecycle event for optional observed tools."""

        call_id: str
        name: str
        phase: Literal["started", "returned"]
        argument_keys: tuple[str, ...] = ()
        elapsed_s: float | None = None
        error: Any = None
        failed: bool = False
        cancelled: bool = False
        category: str = "generic"


try:
    from rpent.planner.base import DEFAULT_REASONING_EFFORT
except ImportError:
    # Preserve eva's current RPent default when the newer constant is absent.
    DEFAULT_REASONING_EFFORT = "none"


class CompatibleDashboardEventSink:
    """Forward events and downgrade newer optional event types when needed."""

    def __init__(self, inner: DashboardEventSink) -> None:
        self._inner = inner

    @property
    def enabled(self) -> bool:
        return bool(self._inner.enabled)

    def emit(self, event: Any) -> None:
        try:
            self._inner.emit(event)
        except TypeError:
            # The current Dashboard state only accepts its original event
            # union. Detailed planner events are optional projections; a
            # native UsageEvent preserves aggregate counters, while context
            # and catalog events are safely omitted.
            if isinstance(event, PlannerRequestUsageEvent):
                self._inner.emit(
                    UsageEvent(
                        inp=int(event.input_tokens),
                        out=int(event.output_tokens),
                        tool_calls=0,
                    )
                )
                return
            if isinstance(
                event,
                (CapabilityCatalogEvent, PlannerSystemContextEvent, ToolCallEvent),
            ):
                return
            raise


def adapt_dashboard_events(
    dashboard_events: DashboardEventSink | None,
) -> CompatibleDashboardEventSink:
    """Return one sink adapter without creating a second event owner."""
    return CompatibleDashboardEventSink(
        dashboard_events or NullDashboardEventSink()
    )


def dashboard_catalog_for_toolkit(toolkit: Any) -> tuple[CapabilityCatalogEntry, ...]:
    """Use a Toolkit-owned catalog when available, otherwise derive one."""
    catalog = getattr(toolkit, "dashboard_catalog", None)
    if callable(catalog):
        return tuple(catalog())
    entries: list[CapabilityCatalogEntry] = []
    for spec in toolkit.get_tools_spec():
        if not isinstance(spec, Mapping):
            continue
        name = spec.get("name")
        if not isinstance(name, str) or not name:
            continue
        schema = spec.get("input_schema")
        properties = (
            tuple(str(key) for key in schema.get("properties", {}))
            if isinstance(schema, Mapping)
            else ()
        )
        description = spec.get("description", "")
        entries.append(
            CapabilityCatalogEntry(
                tool_name=name,
                summary=" ".join(str(description).split()),
                parameter_names=properties,
                source="toolkit",
            )
        )
    return tuple(entries)


def execute_observed_tool(
    *,
    dashboard_events: DashboardEventSink,
    name: str,
    input_dict: dict[str, Any],
    category: str,
    handler: Any,
) -> ToolResult:
    """Execute an optional read-only tool and return the current ToolResult."""
    call_id = uuid.uuid4().hex
    argument_keys = tuple(key for key in input_dict if isinstance(key, str))

    def publish(event: Any) -> None:
        try:
            if dashboard_events.enabled:
                dashboard_events.emit(event)
        except Exception:
            # Observability must not change tool execution semantics.
            return

    publish(
        ToolCallEvent(
            call_id=call_id,
            name=name,
            phase="started",
            argument_keys=argument_keys,
            category=category,
        )
    )
    started = time.perf_counter()
    tool_result: ToolResult | None = None
    raised_error: BaseException | None = None
    try:
        result = handler(name, input_dict)
        if not isinstance(result, ToolResult):
            result = ToolResult(name=name, result=dict(result))
        result.call_id = call_id
        tool_result = result
        return result
    except Exception as exc:  # noqa: BLE001 - tool boundary
        raised_error = exc
        result = {
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        tool_result = ToolResult(name=name, result=result, call_id=call_id)
        return tool_result
    finally:
        payload = (
            tool_result.result
            if tool_result is not None and isinstance(tool_result.result, Mapping)
            else {}
        )
        result_error = payload.get("error")
        publish(
            ToolCallEvent(
                call_id=call_id,
                name=name,
                phase="returned",
                argument_keys=argument_keys,
                elapsed_s=round(time.perf_counter() - started, 3),
                error=raised_error if raised_error is not None else result_error,
                failed=raised_error is not None or result_error is not None,
                category=category,
            )
        )


__all__ = [
    "CapabilityCatalogEntry",
    "CapabilityCatalogEvent",
    "CompatibleDashboardEventSink",
    "DashboardEventSink",
    "DEFAULT_REASONING_EFFORT",
    "NullDashboardEventSink",
    "PlannerRequestUsageEvent",
    "PlannerResult",
    "PlannerSystemContextEvent",
    "ToolCallEvent",
    "TranscriptEvent",
    "UsageEvent",
    "adapt_dashboard_events",
    "dashboard_catalog_for_toolkit",
    "execute_observed_tool",
]
