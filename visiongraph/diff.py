"""Compares two VisualState observations - added/removed/unchanged objects
and per-object state transitions (e.g. a calendar day going from
unselected to selected).

Object identity heuristic: (type, label). V0.1 has no entity tracking (see
PROJECT_SUMMARY.md), so this is a documented simplification, not real
re-identification - if two distinct physical objects ever shared the same
(type, label) across observations, this would read as one object changing
state rather than one disappearing and a different one appearing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from visiongraph.state import DetectedObject, VisualState

ObjectKey = Tuple[str, str]  # (type, label)


@dataclass(frozen=True)
class ObjectStateChange:
    type: str
    label: str
    before_state: str | None
    after_state: str | None


@dataclass(frozen=True)
class VisualStateDiff:
    added: List[DetectedObject]
    removed: List[DetectedObject]
    unchanged: List[DetectedObject]
    state_changes: List[ObjectStateChange]


def _key(obj: DetectedObject) -> ObjectKey:
    return (obj.type, obj.label)


def diff_visual_states(before: VisualState, after: VisualState) -> VisualStateDiff:
    """same (type, label) + same state     -> unchanged
    same (type, label) + different state  -> state_changes
    only in after                         -> added
    only in before                        -> removed
    """
    before_by_key: Dict[ObjectKey, DetectedObject] = {_key(obj): obj for obj in before.objects}
    after_by_key: Dict[ObjectKey, DetectedObject] = {_key(obj): obj for obj in after.objects}

    added = [obj for key, obj in after_by_key.items() if key not in before_by_key]
    removed = [obj for key, obj in before_by_key.items() if key not in after_by_key]

    unchanged: List[DetectedObject] = []
    state_changes: List[ObjectStateChange] = []
    for key, after_obj in after_by_key.items():
        before_obj = before_by_key.get(key)
        if before_obj is None:
            continue
        if before_obj.state != after_obj.state:
            state_changes.append(
                ObjectStateChange(
                    type=key[0],
                    label=key[1],
                    before_state=before_obj.state,
                    after_state=after_obj.state,
                )
            )
        else:
            unchanged.append(after_obj)

    return VisualStateDiff(added=added, removed=removed, unchanged=unchanged, state_changes=state_changes)
