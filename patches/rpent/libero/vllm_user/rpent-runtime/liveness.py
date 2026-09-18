# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Provider-neutral liveness policy for RPent tool calls.

The policy understands only the common ``Toolkit`` result envelope. Robot
transitions, camera roles, and recovery motions remain owned by the robot
Toolkit and are surfaced through its structured ``code`` and ``instruction``
fields. This keeps model adapters and the shared planner runtime free of
embodiment-specific rules.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

ParsedToolCall = tuple[str, str, Mapping[str, object]]


@dataclass(slots=True)
class _ToolOutcome:
    fingerprint: str
    repeats: int
    environment_observation: bool = False


@dataclass(slots=True)
class ToolLoopGuard:
    """Track repeated tool outcomes and observable environment progress."""

    enabled: bool = True
    max_identical_errors: int = 3
    max_consecutive_errors: int = 12
    max_identical_outcomes: int = 4
    max_repeated_failure_families: int = 2
    max_observations_without_action: int = 16
    max_stagnant_actions: int = 4
    max_consecutive_identical_actions: int = 2
    max_equivalent_capped_actions: int = 8
    state_tolerance: float = 0.005
    _last_by_call: dict[str, _ToolOutcome] = field(default_factory=dict, init=False)
    _last_error_by_call: dict[str, _ToolOutcome] = field(
        default_factory=dict, init=False
    )
    _failure_family_repeats: dict[tuple[str, str], int] = field(
        default_factory=dict, init=False
    )
    _failure_family_scopes: dict[tuple[str, str], object] = field(
        default_factory=dict, init=False
    )
    _failed_call_recovery: dict[str, dict[str, object]] = field(
        default_factory=dict, init=False
    )
    _equivalent_capped_actions: dict[str, int] = field(default_factory=dict, init=False)
    _last_action_call: str | None = field(default=None, init=False)
    _consecutive_identical_actions: int = field(default=0, init=False)
    _max_consecutive_identical_actions_seen: int = field(default=0, init=False)
    total_errors: int = field(default=0, init=False)
    consecutive_errors: int = field(default=0, init=False)
    observations_without_action: int = field(default=0, init=False)
    max_observations_seen: int = field(default=0, init=False)
    stagnant_actions: int = field(default=0, init=False)
    state_changes: int = field(default=0, init=False)
    semantic_feedbacks: int = field(default=0, init=False)
    loop_feedbacks: int = field(default=0, init=False)
    _last_step: float | None = field(default=None, init=False)
    _state_fingerprint: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        for name, value in (
            ("max_identical_errors", self.max_identical_errors),
            ("max_consecutive_errors", self.max_consecutive_errors),
            ("max_identical_outcomes", self.max_identical_outcomes),
            ("max_repeated_failure_families", self.max_repeated_failure_families),
            ("max_observations_without_action", self.max_observations_without_action),
            ("max_stagnant_actions", self.max_stagnant_actions),
            (
                "max_consecutive_identical_actions",
                self.max_consecutive_identical_actions,
            ),
            ("max_equivalent_capped_actions", self.max_equivalent_capped_actions),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if (
            isinstance(self.state_tolerance, bool)
            or not isinstance(self.state_tolerance, (int, float))
            or not math.isfinite(self.state_tolerance)
            or self.state_tolerance <= 0
        ):
            raise ValueError("state_tolerance must be positive")

    def validate_proposal(
        self,
        calls: Sequence[ParsedToolCall],
    ) -> dict[str, object] | None:
        """Reject an exact deterministic failure or a third unchanged call."""
        if not self.enabled:
            return None
        for index, (_call_id, tool, arguments) in enumerate(calls):
            call_key = _call_key(tool, arguments)
            recovery = self._failed_call_recovery.get(call_key)
            if recovery is not None:
                feedback = {
                    "schema": "rpent.planner_repair_feedback.v1",
                    "stage": "planner_call",
                    "path": f"/tool_calls/{index}",
                    "code": "repeated_failed_tool_call",
                    "expected": recovery["instruction"],
                    "actual": f"unchanged failed proposal for {tool}",
                    "repairable": True,
                }
                if "suggested_call" in recovery:
                    feedback["suggested_call"] = recovery["suggested_call"]
                return feedback
            if (
                call_key == self._last_action_call
                and self._consecutive_identical_actions
                >= self.max_consecutive_identical_actions
            ):
                return {
                    "schema": "rpent.planner_repair_feedback.v1",
                    "stage": "planner_call",
                    "path": f"/tool_calls/{index}",
                    "code": "repeated_action_without_replan",
                    "expected": (
                        "this exact state-changing call has already executed "
                        f"{self._consecutive_identical_actions} consecutive times "
                        "without native task success; use the fresh result to "
                        "choose a materially different action, or finish with "
                        "status='stuck' when no safe recovery remains"
                    ),
                    "actual": f"unchanged action proposal for {tool}",
                    "repairable": True,
                }
            previous = self._last_by_call.get(call_key)
            if previous is None or previous.repeats < 2:
                continue
            return {
                "schema": "rpent.planner_repair_feedback.v1",
                "stage": "planner_call",
                "path": f"/tool_calls/{index}",
                "code": "repeated_tool_call_no_progress",
                "expected": (
                    "use the existing tool result to advance the plan, or call "
                    "a different tool with materially different arguments"
                ),
                "actual": f"third unchanged proposal for {tool}",
                "repairable": True,
            }
        return None

    def observe(
        self,
        *,
        tool: str,
        arguments: Mapping[str, object],
        result: Mapping[str, object],
        result_text: str | None = None,
        result_fingerprint: str | None = None,
        failure_scope: object | None = None,
    ) -> tuple[dict[str, object] | None, str | None]:
        """Return structured feedback and an optional terminal liveness error."""
        if result.get("_finish") is True:
            return None, None

        error = result.get("error")
        if error:
            self.total_errors += 1
            self.consecutive_errors += 1
        else:
            self.consecutive_errors = 0
        if not self.enabled:
            return None, None

        call_key = _call_key(tool, arguments)
        outcome = (
            _failure_outcome(result)
            if error
            else result_fingerprint
            if result_fingerprint is not None
            else result_text
            if result_text is not None
            else dict(result)
        )
        fingerprint = _fingerprint(outcome)
        outcomes = self._last_error_by_call if error else self._last_by_call
        previous = outcomes.get(call_key)
        repeats = (
            previous.repeats + 1
            if previous and previous.fingerprint == fingerprint
            else 1
        )
        outcomes[call_key] = _ToolOutcome(fingerprint, repeats)

        if error:
            recovery = _failure_recovery(result)
            if _is_deterministic_failure(result):
                self._failed_call_recovery[call_key] = recovery
            family_key = _failure_family_key(tool, result)
            family_repeats = self._failure_family_repeats.get(family_key, 0)
            # Several independent read-only calls can belong to one planner
            # proposal. Count that proposal once so feedback is visible before
            # the cross-proposal guard terminates the failure family.
            if (
                failure_scope is None
                or self._failure_family_scopes.get(family_key) != failure_scope
            ):
                family_repeats += 1
                self._failure_family_repeats[family_key] = family_repeats
                if failure_scope is not None:
                    self._failure_family_scopes[family_key] = failure_scope
            feedback: dict[str, object] = {
                "schema": "rpent.tool_execution_feedback.v1",
                "stage": "tool_execution",
                "tool": tool,
                "error": str(error),
                "identical_failure_count": repeats,
                "instruction": recovery["instruction"],
            }
            if "suggested_call" in recovery:
                feedback["suggested_call"] = recovery["suggested_call"]
            if family_repeats >= self.max_repeated_failure_families:
                self.semantic_feedbacks += 1
                feedback.update(
                    {
                        "code": "repeated_failure_family",
                        "failure_family_count": family_repeats,
                        "instruction": (
                            "The same semantic failure persists despite changed "
                            "arguments. Do not retry or reword the same proposal; "
                            "follow the tool-provided recovery, choose a materially "
                            "different action, or finish with status='stuck'."
                        ),
                    }
                )
                if family_repeats > self.max_repeated_failure_families:
                    self.loop_feedbacks += 1
                    return feedback, (
                        "tool loop guard stopped after "
                        f"{family_repeats} repeated failures in one semantic "
                        f"failure family for {tool}"
                    )
            if repeats >= self.max_identical_errors:
                self.loop_feedbacks += 1
                return feedback, (
                    f"tool loop guard stopped {repeats} identical failures for {tool}"
                )
            if self.consecutive_errors >= self.max_consecutive_errors:
                self.loop_feedbacks += 1
                return feedback, (
                    "tool loop guard stopped after "
                    f"{self.consecutive_errors} consecutive execution errors"
                )
            self.loop_feedbacks += 1
            return feedback, None

        if tool == "reset":
            self._failure_family_repeats.clear()
            self._failure_family_scopes.clear()
        else:
            self._failure_family_repeats = {
                key: count
                for key, count in self._failure_family_repeats.items()
                if key[0] != tool
            }
            self._failure_family_scopes = {
                key: scope
                for key, scope in self._failure_family_scopes.items()
                if key[0] != tool
            }

        progress_feedback, progress_error = self._observe_progress(
            tool=tool,
            arguments=arguments,
            result=result,
        )
        capped_feedback, capped_error = self._observe_equivalent_capped_action(
            tool=tool,
            result=result,
        )
        feedback = progress_feedback or capped_feedback
        loop_error = progress_error or capped_error
        if feedback is not None:
            self.loop_feedbacks += 1
            return feedback, loop_error

        if repeats >= 2:
            feedback = {
                "schema": "rpent.tool_loop_feedback.v1",
                "stage": "tool_execution",
                "tool": tool,
                "identical_outcome_count": repeats,
                "instruction": (
                    "This identical call produced an unchanged result. Use the "
                    "existing result to advance the plan or choose a different call."
                ),
            }
            self.loop_feedbacks += 1
            if repeats >= self.max_identical_outcomes:
                return feedback, (
                    f"tool loop guard stopped {repeats} identical outcomes for {tool}"
                )
            return feedback, None
        return None, None

    def _observe_progress(
        self,
        *,
        tool: str,
        arguments: Mapping[str, object],
        result: Mapping[str, object],
    ) -> tuple[dict[str, object] | None, str | None]:
        step = _environment_step(result)
        previous_state = self._state_fingerprint
        current_state = _environment_state_fingerprint(
            result, tolerance=float(self.state_tolerance)
        )
        if current_state is not None:
            self._state_fingerprint = current_state
        state_changed = (
            previous_state is not None
            and current_state is not None
            and current_state != previous_state
        )
        is_action = _is_environment_action(
            tool=tool,
            result=result,
            step=step,
            previous_step=self._last_step,
        )
        if step is not None and (self._last_step is None or step > self._last_step):
            self._last_step = step

        if is_action:
            # A mutation makes earlier environment reads stale, even when the
            # next request uses the same latest-step alias. Static file reads
            # and action outcomes retain their independent loop protection.
            self._last_by_call = {
                key: outcome
                for key, outcome in self._last_by_call.items()
                if not outcome.environment_observation
            }
            self._failed_call_recovery.clear()
            self._last_error_by_call.clear()
            self.observations_without_action = 0
            progress = _planner_progress(result, source_tool=tool)
            if progress is not None and progress.get("native_success") is True:
                self._last_action_call = None
                self._consecutive_identical_actions = 0
                self.stagnant_actions = 0
                if state_changed:
                    self.state_changes += 1
                return None, None
            self._record_action_call(_call_key(tool, arguments))
            if previous_state is None or current_state is None:
                self.stagnant_actions = 0
                return None, None
            if state_changed:
                self.stagnant_actions = 0
                self.state_changes += 1
                return None, None
            self.stagnant_actions += 1
            self.semantic_feedbacks += 1
            recovery = _result_progress_recovery(result, source_tool=tool)
            if recovery is None:
                recovery = {
                    "instruction": (
                        "The observable environment state did not measurably "
                        "change after this action. Re-observe and materially "
                        "revise the action rather than repeating an equivalent "
                        "call."
                    )
                }
            self._failed_call_recovery[_call_key(tool, arguments)] = recovery
            feedback = {
                "schema": "rpent.semantic_progress_feedback.v1",
                "stage": "tool_execution",
                "tool": tool,
                "unchanged_action_count": self.stagnant_actions,
                "instruction": recovery["instruction"],
            }
            if "suggested_call" in recovery:
                feedback["suggested_call"] = recovery["suggested_call"]
            if self.stagnant_actions >= self.max_stagnant_actions:
                return feedback, (
                    "tool loop guard stopped after "
                    f"{self.stagnant_actions} state-unchanged actions"
                )
            return feedback, None

        # File and memory tools do not necessarily carry environment evidence.
        # Only count observations tied to an environment step or state snapshot.
        if step is None and current_state is None:
            return None, None
        self._last_by_call[_call_key(tool, arguments)].environment_observation = True
        self.observations_without_action += 1
        self.max_observations_seen = max(
            self.max_observations_seen, self.observations_without_action
        )
        if self.observations_without_action % self.max_observations_without_action:
            return None, None
        self.semantic_feedbacks += 1
        return (
            {
                "schema": "rpent.semantic_progress_feedback.v1",
                "stage": "tool_execution",
                "tool": tool,
                "observations_without_action": self.observations_without_action,
                "instruction": (
                    "You have gathered many environment observations without "
                    "changing the environment. Use the decisive evidence for the "
                    "next safe action, or state what genuinely new evidence is needed."
                ),
            },
            None,
        )

    def _record_action_call(self, call_key: str) -> None:
        """Track exact consecutive mutations independently of state noise."""
        if call_key == self._last_action_call:
            self._consecutive_identical_actions += 1
        else:
            self._last_action_call = call_key
            self._consecutive_identical_actions = 1
        self._max_consecutive_identical_actions_seen = max(
            self._max_consecutive_identical_actions_seen,
            self._consecutive_identical_actions,
        )

    def _observe_equivalent_capped_action(
        self,
        *,
        tool: str,
        result: Mapping[str, object],
    ) -> tuple[dict[str, object] | None, str | None]:
        progress = _planner_progress(result, source_tool=tool)
        if progress is None:
            return None, None
        if (
            progress.get("outcome") != "budget_exhausted"
            or progress.get("native_success") is True
        ):
            return None, None
        action_family = progress.get("action_family")
        if not isinstance(action_family, str) or not action_family.strip():
            return None, None
        key = " ".join(action_family.casefold().split())
        count = self._equivalent_capped_actions.get(key, 0) + 1
        self._equivalent_capped_actions[key] = count
        if count < 2:
            return None, None
        self.semantic_feedbacks += 1
        feedback = {
            "schema": "rpent.semantic_progress_feedback.v1",
            "stage": "tool_execution",
            "code": "equivalent_capped_action",
            "action_family": action_family,
            "equivalent_capped_action_count": count,
            "instruction": (
                "This action family again exhausted its execution budget without "
                "native task success. Rewording the same family is not a different "
                "strategy; inspect current evidence and change the setup or action "
                "family before retrying."
            ),
        }
        if count >= self.max_equivalent_capped_actions:
            return feedback, (
                "tool loop guard stopped after "
                f"{count} capped executions for one equivalent learned action"
            )
        return feedback, None

    def stats(self) -> dict[str, int]:
        """Return run-local counters with provider-independent names."""
        return {
            "tool_execution_errors": self.total_errors,
            "loop_feedbacks": self.loop_feedbacks,
            "max_observations_without_action": self.max_observations_seen,
            "stagnant_actions": self.stagnant_actions,
            "state_changes": self.state_changes,
            "semantic_progress_feedbacks": self.semantic_feedbacks,
            "max_consecutive_identical_actions_seen": (
                self._max_consecutive_identical_actions_seen
            ),
            "max_equivalent_capped_actions_seen": max(
                self._equivalent_capped_actions.values(), default=0
            ),
        }

    def reset(self) -> None:
        """Clear all state before a planner solve."""
        self._last_by_call.clear()
        self._last_error_by_call.clear()
        self._failure_family_repeats.clear()
        self._failure_family_scopes.clear()
        self._failed_call_recovery.clear()
        self._equivalent_capped_actions.clear()
        self._last_action_call = None
        self._consecutive_identical_actions = 0
        self._max_consecutive_identical_actions_seen = 0
        self.total_errors = 0
        self.consecutive_errors = 0
        self.observations_without_action = 0
        self.max_observations_seen = 0
        self.stagnant_actions = 0
        self.state_changes = 0
        self.semantic_feedbacks = 0
        self.loop_feedbacks = 0
        self._last_step = None
        self._state_fingerprint = None


def tool_loop_guard_from_env(
    environ: Mapping[str, str] | None = None,
) -> ToolLoopGuard:
    """Build the shared guard, accepting legacy vLLM names as fallbacks."""
    env = os.environ if environ is None else environ
    mode = (
        _env_value(
            env,
            "RPENT_PLANNER_LOOP_GUARD_MODE",
            "RPENT_VLLM_LOOP_GUARD_MODE",
            default="enabled",
        )
        .strip()
        .lower()
    )
    if mode not in {"enabled", "disabled"}:
        raise ValueError(
            "RPENT_PLANNER_LOOP_GUARD_MODE must be 'enabled' or 'disabled'"
        )
    return ToolLoopGuard(
        enabled=mode == "enabled",
        max_identical_errors=_env_int(
            env,
            "RPENT_PLANNER_MAX_IDENTICAL_TOOL_ERRORS",
            "RPENT_VLLM_MAX_IDENTICAL_TOOL_ERRORS",
            default=3,
        ),
        max_consecutive_errors=_env_int(
            env,
            "RPENT_PLANNER_MAX_CONSECUTIVE_TOOL_ERRORS",
            "RPENT_VLLM_MAX_CONSECUTIVE_TOOL_ERRORS",
            legacy_name="RPENT_VLLM_MAX_TOTAL_TOOL_ERRORS",
            default=12,
        ),
        max_identical_outcomes=_env_int(
            env,
            "RPENT_PLANNER_MAX_IDENTICAL_TOOL_OUTCOMES",
            "RPENT_VLLM_MAX_IDENTICAL_TOOL_OUTCOMES",
            default=4,
        ),
        max_repeated_failure_families=_env_int(
            env,
            "RPENT_PLANNER_MAX_REPEATED_FAILURE_FAMILIES",
            "RPENT_VLLM_MAX_REPEATED_FAILURE_FAMILIES",
            default=2,
        ),
        max_observations_without_action=_env_int(
            env,
            "RPENT_PLANNER_MAX_OBSERVATIONS_WITHOUT_ACTION",
            "RPENT_VLLM_MAX_OBSERVATIONS_WITHOUT_ACTION",
            default=16,
        ),
        max_stagnant_actions=_env_int(
            env,
            "RPENT_PLANNER_MAX_STAGNANT_ACTIONS",
            "RPENT_VLLM_MAX_STAGNANT_ACTIONS",
            default=4,
        ),
        max_consecutive_identical_actions=_env_int(
            env,
            "RPENT_PLANNER_MAX_CONSECUTIVE_IDENTICAL_ACTIONS",
            "RPENT_VLLM_MAX_CONSECUTIVE_IDENTICAL_ACTIONS",
            default=2,
        ),
        max_equivalent_capped_actions=_env_int(
            env,
            "RPENT_PLANNER_MAX_EQUIVALENT_CAPPED_ACTIONS",
            "RPENT_VLLM_MAX_EQUIVALENT_CAPPED_ACTIONS",
            default=8,
        ),
        state_tolerance=_env_float(
            env,
            "RPENT_PLANNER_STATE_TOLERANCE",
            "RPENT_VLLM_STATE_TOLERANCE",
            default=0.005,
        ),
    )


def tool_result_fingerprint(result: Mapping[str, object], blocks: object) -> str:
    """Fingerprint semantic text and images without retaining image payloads."""
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            _semantic_tool_value(result),
            ensure_ascii=False,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for block in blocks or ():
        if not isinstance(block, Mapping) or block.get("type") != "image":
            continue
        source = block.get("source")
        if not isinstance(source, Mapping):
            continue
        data = source.get("data")
        if isinstance(data, str):
            digest.update(hashlib.sha256(data.encode("ascii")).digest())
    return digest.hexdigest()


def _call_key(tool: str, arguments: Mapping[str, object]) -> str:
    return json.dumps(
        {"tool": tool, "arguments": dict(arguments)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _failure_outcome(result: Mapping[str, object]) -> dict[str, object]:
    keys = ("error", "code", "reason", "instruction", "fallback")
    return {key: result[key] for key in keys if result.get(key) is not None}


def _failure_family_key(tool: str, result: Mapping[str, object]) -> tuple[str, str]:
    code = result.get("code")
    failure = (
        {"code": code.strip()}
        if isinstance(code, str) and code.strip()
        else {"error": str(result.get("error", ""))}
    )
    return tool, json.dumps(
        {"failure": failure},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _failure_recovery(result: Mapping[str, object]) -> dict[str, object]:
    instruction = result.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        fallback = result.get("fallback")
        if isinstance(fallback, str) and fallback.strip():
            instruction = f"Follow the tool-provided fallback: {fallback.strip()}"
        else:
            instruction = (
                "Correct the arguments or choose a materially different tool call. "
                "Do not repeat this unchanged failure."
            )
    recovery: dict[str, object] = {"instruction": instruction}
    suggested = result.get("suggested_call")
    if isinstance(suggested, Mapping):
        recovery["suggested_call"] = dict(suggested)
    return recovery


def _is_deterministic_failure(result: Mapping[str, object]) -> bool:
    code = result.get("code")
    return isinstance(code, str) and bool(code.strip())


def _result_progress_recovery(
    result: Mapping[str, object],
    *,
    source_tool: str,
) -> dict[str, object] | None:
    """Project owner-supplied progress guidance without interpreting it."""
    progress = _planner_progress(result, source_tool=source_tool)
    instruction = progress.get("instruction") if progress is not None else None
    if not isinstance(instruction, str) or not instruction.strip():
        return None
    recovery: dict[str, object] = {"instruction": instruction}
    suggested = progress.get("suggested_call")
    if isinstance(suggested, Mapping):
        recovery["suggested_call"] = dict(suggested)
    return recovery


def _planner_progress(
    result: Mapping[str, object], *, source_tool: str
) -> Mapping[str, object] | None:
    """Return one current-call common progress envelope, ignoring stale data."""
    progress = result.get("planner_progress")
    if not isinstance(progress, Mapping):
        return None
    if progress.get("schema") != "rpent.planner_progress.v1":
        return None
    if progress.get("source_tool") != source_tool:
        return None
    return progress


def _environment_step(result: Mapping[str, object]) -> float | None:
    for candidate in (result, result.get("state")):
        if not isinstance(candidate, Mapping):
            continue
        for key in ("step", "step_idx"):
            value = candidate.get(key)
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            ):
                return float(value)
    return None


def _is_environment_action(
    *,
    tool: str,
    result: Mapping[str, object],
    step: float | None,
    previous_step: float | None,
) -> bool:
    progress = _planner_progress(result, source_tool=tool)
    if progress is not None and progress.get("outcome") == "observed":
        # A robot Toolkit may need to persist a fresh observation as a new
        # EnvState step. The explicit owner projection takes precedence over
        # step/command heuristics so observation capture is not treated as a
        # state-changing robot action.
        return False
    elapsed = result.get("agent_elapsed_s")
    if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool):
        return True
    if step is not None and previous_step is not None and step > previous_step:
        return True
    command = result.get("command")
    if not isinstance(command, Mapping):
        log = result.get("log")
        command = log.get("command") if isinstance(log, Mapping) else None
    return isinstance(command, Mapping) and command.get("action") == tool


_VOLATILE_STATE_KEYS = {
    "agent_elapsed_s",
    "artifacts",
    "elapsed_s",
    "images",
    "log",
    "native_actions",
    "policy_actions",
    "step",
    "step_idx",
    "steps_used",
    "take_action_cnt",
    "task_language",
    "view_specs",
}


def _environment_state_fingerprint(
    result: Mapping[str, object], *, tolerance: float
) -> str | None:
    state = result.get("state")
    snapshot: dict[str, object] = {}
    if isinstance(state, Mapping) and state:
        snapshot["state"] = state
    for key in (
        "success",
        "terminated",
        "truncated",
    ):
        if key in result:
            snapshot[key] = result[key]
    if not snapshot:
        return None
    return _fingerprint(_stable_state(snapshot, tolerance=tolerance))


def _stable_state(value: object, *, tolerance: float, key: str = "") -> object:
    if key in _VOLATILE_STATE_KEYS or "artifact" in key or key.startswith("image"):
        return None
    if isinstance(value, Mapping):
        return {
            str(item_key): projected
            for item_key, item_value in value.items()
            if (
                projected := _stable_state(
                    item_value, tolerance=tolerance, key=str(item_key)
                )
            )
            is not None
        }
    if isinstance(value, (list, tuple)):
        return [_stable_state(item, tolerance=tolerance) for item in value]
    if isinstance(value, float):
        return round(value / tolerance)
    return value


def _semantic_tool_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            key: _semantic_tool_value(item)
            for key, item in value.items()
            if not str(key).startswith("_image_")
            and key not in {"segment_artifact", "overlay_artifact"}
        }
    if isinstance(value, (list, tuple)):
        return [_semantic_tool_value(item) for item in value]
    return value


def _env_value(
    env: Mapping[str, str],
    name: str,
    fallback_name: str,
    *,
    default: str,
) -> str:
    return env.get(name) or env.get(fallback_name) or default


def _env_int(
    env: Mapping[str, str],
    name: str,
    fallback_name: str,
    *,
    default: int,
    legacy_name: str | None = None,
) -> int:
    raw = env.get(name) or env.get(fallback_name)
    if not raw and legacy_name is not None:
        raw = env.get(legacy_name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _env_float(
    env: Mapping[str, str],
    name: str,
    fallback_name: str,
    *,
    default: float,
) -> float:
    raw = env.get(name) or env.get(fallback_name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


__all__ = [
    "ParsedToolCall",
    "ToolLoopGuard",
    "tool_loop_guard_from_env",
    "tool_result_fingerprint",
]
