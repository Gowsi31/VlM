"""Tests for the diff/verification layer: visiongraph/diff.py and
VerificationNode (visiongraph/nodes.py).
"""

from PIL import Image

from visiongraph.adapters import CallableVLMAdapter, VLMResponse
from visiongraph.diff import ObjectStateChange, VisualStateDiff, diff_visual_states
from visiongraph.graph import VisionGraph
from visiongraph.nodes import ActionNode, DecisionNode, VerificationNode, VisionNode
from visiongraph.state import DetectedObject, GraphState, VisualState

A_PIXEL = Image.new("RGB", (2, 2))


# ---------------------------------------------------------------------------
# diff_visual_states
# ---------------------------------------------------------------------------


class TestDiffVisualStates:
    def test_added_objects_detected(self):
        before = VisualState(objects=[])
        after = VisualState(objects=[DetectedObject(type="ui_element", label="chat bubble")])

        diff = diff_visual_states(before, after)

        assert [o.label for o in diff.added] == ["chat bubble"]
        assert diff.removed == []
        assert diff.unchanged == []
        assert diff.state_changes == []

    def test_removed_objects_detected(self):
        before = VisualState(objects=[DetectedObject(type="ui_element", label="chat bubble")])
        after = VisualState(objects=[])

        diff = diff_visual_states(before, after)

        assert [o.label for o in diff.removed] == ["chat bubble"]
        assert diff.added == []

    def test_unchanged_when_same_identity_and_same_state(self):
        obj_before = DetectedObject(type="ui_element", label="graph", state=None)
        obj_after = DetectedObject(type="ui_element", label="graph", state=None)

        diff = diff_visual_states(VisualState(objects=[obj_before]), VisualState(objects=[obj_after]))

        assert [o.label for o in diff.unchanged] == ["graph"]
        assert diff.added == diff.removed == diff.state_changes == []

    def test_identity_is_type_and_label_only_not_full_object_equality(self):
        # Different confidence/description/bbox, same (type, label), same
        # state -> still "unchanged". (type, label) is the whole identity
        # heuristic, documented as a V0.1 simplification, not real tracking.
        obj_before = DetectedObject(type="ui_element", label="graph", confidence=0.5, description="a chart")
        obj_after = DetectedObject(type="ui_element", label="graph", confidence=0.99, description="")

        diff = diff_visual_states(VisualState(objects=[obj_before]), VisualState(objects=[obj_after]))

        assert [o.label for o in diff.unchanged] == ["graph"]

    def test_calendar_day_selection_state_change(self):
        """The exact scenario that motivated adding DetectedObject.state:
        both day 7 and day 8 exist before and after, but selection moves
        from 7 to 8. A label-only diff cannot see this - state can."""
        before = VisualState(
            objects=[
                DetectedObject(type="calendar", label="7", state="selected"),
                DetectedObject(type="calendar", label="8", state="unselected"),
            ]
        )
        after = VisualState(
            objects=[
                DetectedObject(type="calendar", label="7", state="unselected"),
                DetectedObject(type="calendar", label="8", state="selected"),
            ]
        )

        diff = diff_visual_states(before, after)

        assert diff.added == []
        assert diff.removed == []
        assert diff.unchanged == []
        assert len(diff.state_changes) == 2

        changes_by_label = {c.label: c for c in diff.state_changes}
        assert changes_by_label["7"] == ObjectStateChange(
            type="calendar", label="7", before_state="selected", after_state="unselected"
        )
        assert changes_by_label["8"] == ObjectStateChange(
            type="calendar", label="8", before_state="unselected", after_state="selected"
        )

    def test_state_none_to_value_is_a_state_change(self):
        before = VisualState(objects=[DetectedObject(type="calendar", label="7", state=None)])
        after = VisualState(objects=[DetectedObject(type="calendar", label="7", state="selected")])

        diff = diff_visual_states(before, after)

        assert diff.state_changes == [
            ObjectStateChange(type="calendar", label="7", before_state=None, after_state="selected")
        ]

    def test_state_value_to_none_is_a_state_change(self):
        before = VisualState(objects=[DetectedObject(type="calendar", label="7", state="selected")])
        after = VisualState(objects=[DetectedObject(type="calendar", label="7", state=None)])

        diff = diff_visual_states(before, after)

        assert diff.state_changes == [
            ObjectStateChange(type="calendar", label="7", before_state="selected", after_state=None)
        ]

    def test_realistic_ui_diff_with_added_element_and_unchanged_rest(self):
        """Mirrors the earlier Colab prototype's exact result: a dialog
        appears, everything else is unchanged."""
        labels = ["chat bubble", "profile picture", "graph", "calendar"]
        before = VisualState(objects=[DetectedObject(type="ui_element", label=l) for l in labels])
        after = VisualState(
            objects=[DetectedObject(type="ui_element", label=l) for l in labels]
            + [DetectedObject(type="ui_element", label="calendar dialog")]
        )

        diff = diff_visual_states(before, after)

        assert [o.label for o in diff.added] == ["calendar dialog"]
        assert diff.removed == []
        assert sorted(o.label for o in diff.unchanged) == sorted(labels)


# ---------------------------------------------------------------------------
# VerificationNode
# ---------------------------------------------------------------------------


class TestVerificationNode:
    def test_first_execution_captures_baseline_with_no_diff(self):
        vs = VisualState(objects=[DetectedObject(type="calendar", label="7", state="selected")])
        node = VerificationNode()

        result = node.execute(GraphState(visual_state=vs))

        assert result.memory["verification_diff"] is None
        assert result.memory["_verification_baseline"] is vs

    def test_second_execution_diffs_against_the_first(self):
        node = VerificationNode()
        vs_a = VisualState(objects=[DetectedObject(type="calendar", label="7", state="selected")])
        vs_b = VisualState(objects=[DetectedObject(type="calendar", label="7", state="unselected")])

        state = node.execute(GraphState(visual_state=vs_a))
        state = node.execute(state.update(visual_state=vs_b))

        diff = state.memory["verification_diff"]
        assert diff.state_changes == [
            ObjectStateChange(type="calendar", label="7", before_state="selected", after_state="unselected")
        ]

    def test_baseline_rolls_forward_each_execution(self):
        """Third execution compares against the SECOND observation, not
        the first - the baseline is the most recent one seen, not fixed."""
        node = VerificationNode()
        vs_a = VisualState(objects=[DetectedObject(type="calendar", label="7", state="selected")])
        vs_b = VisualState(objects=[DetectedObject(type="calendar", label="7", state="unselected")])
        vs_c = VisualState(objects=[DetectedObject(type="calendar", label="7", state="unselected")])

        state = node.execute(GraphState(visual_state=vs_a))  # baseline = A, diff = None
        state = node.execute(state.update(visual_state=vs_b))  # baseline = B, diff = A vs B
        state = node.execute(state.update(visual_state=vs_c))  # baseline = C, diff = B vs C

        # B -> C: no change, since both are "unselected"
        diff = state.memory["verification_diff"]
        assert diff.state_changes == []
        assert [o.label for o in diff.unchanged] == ["7"]

    def test_logs_a_step_with_the_verification_summary(self):
        vs = VisualState(objects=[])
        node = VerificationNode()

        result = node.execute(GraphState(visual_state=vs))

        assert result.history[-1].node_name == "VerificationNode"
        assert "baseline captured" in result.history[-1].summary


# ---------------------------------------------------------------------------
# Full loop: Vision -> Verify -> Decide -> Action -> Vision -> Verify -> stop
#
# Reproduces the calendar day-selection scenario end-to-end: click day 8,
# re-observe, and confirm via VerificationNode's diff (not a label-only
# comparison) that day 8 actually became selected and day 7 did not
# remain selected.
# ---------------------------------------------------------------------------


class TestVerificationIntegration:
    def test_click_day_8_verified_via_state_change_diff(self):
        calls = {"n": 0}

        def fake_generate(image, prompt):
            calls["n"] += 1
            if calls["n"] == 1:
                objects = [
                    {"type": "calendar", "label": "7", "state": "selected"},
                    {"type": "calendar", "label": "8", "state": "unselected"},
                ]
            else:
                objects = [
                    {"type": "calendar", "label": "7", "state": "unselected"},
                    {"type": "calendar", "label": "8", "state": "selected"},
                ]
            return VLMResponse(
                text="calendar view",
                structured={"scene": "calendar view", "objects": objects},
            )

        def decide_logic(state):
            diff = state.memory.get("verification_diff")
            if diff is None:
                return "action"  # first observation - always act once

            day_8_selected = any(
                o.type == "calendar" and o.label == "8" and o.state == "selected"
                for o in state.visual_state.objects
            )
            return None if day_8_selected else "action"  # stop once verified, else retry

        graph = VisionGraph(name="calendar-click-verify")
        graph.add_node(
            "vision",
            VisionNode(
                prompt="What's the calendar state?",
                adapter=CallableVLMAdapter(fake_generate),
                structured_output=True,
            ),
        )
        graph.add_node("verify", VerificationNode())
        graph.add_node("decide", DecisionNode(logic=decide_logic))
        graph.add_node("action", ActionNode(action_type="click_day_8"))
        graph.connect("vision", "verify")
        graph.connect("verify", "decide")
        graph.connect("action", "vision")  # loop back: re-observe after acting
        graph = graph.compile()

        final = graph.run(image=A_PIXEL, prompt="What's the calendar state?", max_steps=10)

        assert calls["n"] == 2  # observed once before the click, once after
        assert [h.node_name for h in final.history] == [
            "VisionNode",
            "VerificationNode",
            "DecisionNode",
            "ActionNode",
            "VisionNode",
            "VerificationNode",
            "DecisionNode",
        ]
        assert len(final.actions_taken) == 1  # clicked exactly once, not looping forever

        diff = final.memory["verification_diff"]
        changes_by_label = {c.label: c for c in diff.state_changes}
        assert changes_by_label["7"].before_state == "selected"
        assert changes_by_label["7"].after_state == "unselected"
        assert changes_by_label["8"].before_state == "unselected"
        assert changes_by_label["8"].after_state == "selected"
        assert final.step_count < 10  # stopped on its own, didn't hit max_steps
