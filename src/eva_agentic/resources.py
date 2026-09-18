"""Host-local resource slots with advisory file locks."""

from __future__ import annotations

import fcntl
import os
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ResourceGrant:
    kind: str
    slot: int
    lock_path: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "slot": self.slot,
            "lock_path": str(self.lock_path),
        }


@dataclass
class _HeldGrant:
    grant: ResourceGrant
    fd: int


@dataclass
class ResourceLease:
    grants: tuple[ResourceGrant, ...]
    _held: tuple[_HeldGrant, ...] = field(default_factory=tuple)

    def release(self) -> None:
        for held in reversed(self._held):
            try:
                fcntl.flock(held.fd, fcntl.LOCK_UN)
            finally:
                os.close(held.fd)
        self._held = ()

    def __enter__(self) -> "ResourceLease":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()


class ResourceAllocator:
    """Allocate preconfigured slots using host-local exclusive file locks."""

    def __init__(self, slots: Mapping[str, Sequence[int]], lock_root: str | Path):
        self.slots = {
            kind: tuple(slots[kind]) for kind in sorted(slots)
        }
        self.lock_root = Path(lock_root)
        for kind, values in self.slots.items():
            if not values or len(set(values)) != len(values):
                raise ValueError(f"resource slots for {kind} must be non-empty and unique")
            if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
                raise ValueError(f"resource slots for {kind} must be non-negative integers")

    def allocate(
        self,
        requirements: Mapping[str, int],
        *,
        timeout_s: float | None = None,
        poll_interval_s: float = 0.05,
    ) -> ResourceLease:
        unknown = sorted(set(requirements) - set(self.slots))
        if unknown:
            raise ValueError(f"unknown resource kinds: {unknown}")
        for kind, count in requirements.items():
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError(f"resource requirement {kind} must be a positive integer")

        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while True:
            held: list[_HeldGrant] = []
            try:
                for kind in sorted(requirements):
                    for slot in self.slots[kind]:
                        if len([item for item in held if item.grant.kind == kind]) >= requirements[kind]:
                            break
                        grant = self._try_lock(kind, slot)
                        if grant is not None:
                            held.append(grant)
                    if len([item for item in held if item.grant.kind == kind]) < requirements[kind]:
                        raise RuntimeError(f"resource slots exhausted: {kind}")
                return ResourceLease(
                    grants=tuple(item.grant for item in held),
                    _held=tuple(held),
                )
            except RuntimeError:
                ResourceLease(
                    grants=tuple(item.grant for item in held),
                    _held=tuple(held),
                ).release()
                if deadline is None or time.monotonic() >= deadline:
                    raise
                time.sleep(poll_interval_s)
            except Exception:
                ResourceLease(
                    grants=tuple(item.grant for item in held),
                    _held=tuple(held),
                ).release()
                raise

    def _try_lock(self, kind: str, slot: int) -> _HeldGrant | None:
        lock_path = self.lock_root / kind / f"{slot}.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, mode=0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError):
            os.close(fd)
            return None
        grant = ResourceGrant(kind=kind, slot=slot, lock_path=lock_path)
        return _HeldGrant(grant=grant, fd=fd)
