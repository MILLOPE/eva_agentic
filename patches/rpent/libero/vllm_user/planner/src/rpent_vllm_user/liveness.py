# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Compatibility imports for the shared RPent planner liveness policy.

The implementation lives in :mod:`rpent.planner.liveness`; this optional
package no longer owns model- or robot-specific loop semantics.
"""

from rpent.planner.liveness import ToolLoopGuard, tool_loop_guard_from_env

__all__ = ["ToolLoopGuard", "tool_loop_guard_from_env"]
