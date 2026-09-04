"""Tests for the V0.1 core loop: state.py, nodes.py, graph.py."""

import pytest

from visiongraph.state import DetectedObject, GraphState, VisualState
from visiongraph.nodes import (
    ActionNode,
    DecisionNode,
    HumanNode,
    ReasoningNode,
    ToolNode,
    VisionNode,
)
from visiongraph.graph import VisionGraph


# ---------------------------------------------------------------------------
# state.py
# ---------------------------------------------------------------------------


class TestGraphState:
    def test_update_returns_new_instance_without_mutating_original(self):
        state = GraphState(prompt="hello")
        updated = state.update(reasoning="thinking")

        assert state.reasoning == ""
        assert updated.reasoning == "thinking"
        assert updated is not state

    def test_log_step_increments_step_count_and_records_history(self):
        state = GraphState()
        updated = state.log_step("vision", summary="saw a button")

        assert updated.step_count == 1
        assert updated.current_node == "vision"
        assert len(updated.history) == 1
        assert updated.history[0].node_name == "vision"
        assert updated.history[0].summary == "saw a button"

    def test_log_step_is_cumulative_across_calls(self):
        state = GraphState().log_step("a").log_step("b")

        assert state.step_count == 2
        assert [h.node_name for h in state.history] == ["a", "b"]


# ---------------------------------------------------------------------------
# nodes.py
# ---------------------------------------------------------------------------


class TestVisionNode:
    def test_stub_backend_produces_labeled_visual_state(self):
        node = VisionNode(prompt="What do you see?")
        result = node.execute(GraphState())

        assert result.visual_state.scene_description == "[no VLM backend configured]"
        assert result.visual_state.confidence == 0.0
        assert result.history[-1].node_name == "VisionNode"

    def test_custom_infer_fn_returning_dict_is_parsed_into_visual_state(self):
        def fake_infer(image, prompt):
            return {
                "objects": [{"type": "button", "label": "Submit", "confidence": 0.9}],
                "scene": "Login form",
                "text": ["Username", "Password"],
                "confidence": 0.87,
            }

        node = VisionNode(prompt="Describe the scene", infer_fn=fake_infer)
        result = node.execute(GraphState(image="frame.png"))

        vs = result.visual_state
        assert vs.scene_description == "Login form"
        assert vs.confidence == 0.87
        assert vs.text_content == ["Username", "Password"]
        assert len(vs.objects) == 1
        assert isinstance(vs.objects[0], DetectedObject)
        assert vs.objects[0].label == "Submit"

    def test_custom_infer_fn_returning_visual_state_passes_through(self):
        vs = VisualState(scene_description="raw state")
        node = VisionNode(prompt="p", infer_fn=lambda image, prompt: vs)
        result = node.execute(GraphState())

        assert result.visual_state is vs

    def test_invalid_infer_fn_return_type_raises(self):
        node = VisionNode(prompt="p", infer_fn=lambda image, prompt: 42)

        with pytest.raises(TypeError):
            node.execute(GraphState())


class TestReasoningNode:
    def test_stub_backend(self):
        node = ReasoningNode(prompt="What should I do?")
        result = node.execute(GraphState())

        assert result.reasoning == "[no reasoning backend configured]"
        assert result.confidence == 0.0

    def test_custom_reasoning_fn(self):
        node = ReasoningNode(
            prompt="p",
            reasoning_fn=lambda state, prompt: ("click submit", 0.92),
        )
        result = node.execute(GraphState())

        assert result.reasoning == "click submit"
        assert result.confidence == 0.92


class TestActionNode:
    def test_stub_backend_records_action(self):
        node = ActionNode(action_type="click", config={"x": 1, "y": 2})
        result = node.execute(GraphState())

        assert len(result.actions_taken) == 1
        record = result.actions_taken[0]
        assert record["action_type"] == "click"
        assert record["result"]["status"] == "stub"

    def test_custom_action_fn_records_real_result(self):
        calls = []

        def fake_action(state, action_type, config):
            calls.append((action_type, config))
            return {"status": "ok"}

        node = ActionNode(action_type="type", config={"text": "hi"}, action_fn=fake_action)
        result = node.execute(GraphState())

        assert calls == [("type", {"text": "hi"})]
        assert result.actions_taken[0]["result"] == {"status": "ok"}

    def test_actions_accumulate_across_multiple_executions(self):
        node = ActionNode(action_type="wait")
        state = node.execute(GraphState())
        state = node.execute(state)

        assert len(state.actions_taken) == 2


class TestDecisionNode:
    def test_sets_next_node_from_logic(self):
        node = DecisionNode(logic=lambda state: "act")
        result = node.execute(GraphState())

        assert result.next_node == "act"

    def test_logic_receives_current_state(self):
        node = DecisionNode(
            logic=lambda state: "confident" if state.confidence > 0.8 else "unsure"
        )

        assert node.execute(GraphState(confidence=0.9)).next_node == "confident"
        assert node.execute(GraphState(confidence=0.1)).next_node == "unsure"


class TestToolNode:
    def test_stub_backend(self):
        node = ToolNode(tool_type="search")
        result = node.execute(GraphState())

        assert "no tool backend configured" in result.memory["tool:search"]

    def test_custom_tool_fn_stores_result_in_memory(self):
        node = ToolNode(tool_type="search", tool_fn=lambda state: ["result1", "result2"])
        result = node.execute(GraphState())

        assert result.memory["tool:search"] == ["result1", "result2"]


class TestHumanNode:
    def test_defaults_to_auto_approve(self):
        node = HumanNode()
        result = node.execute(GraphState())

        assert result.memory["human_approved"] is True

    def test_custom_approve_fn_can_reject(self):
        node = HumanNode(approve_fn=lambda state: False)
        result = node.execute(GraphState())

        assert result.memory["human_approved"] is False


# ---------------------------------------------------------------------------
# graph.py
# ---------------------------------------------------------------------------


class TestVisionGraph:
    def test_entry_point_defaults_to_first_added_node(self):
        graph = VisionGraph()
        graph.add_node("vision", VisionNode(prompt="p"))

        assert graph.entry_point == "vision"

    def test_validate_rejects_edge_to_unknown_node(self):
        graph = VisionGraph()
        graph.add_node("vision", VisionNode(prompt="p"))
        graph.connect("vision", "does_not_exist")

        assert graph.validate() is False
        with pytest.raises(ValueError):
            graph.compile()

    def test_full_loop_runs_end_to_end_in_order(self):
        graph = VisionGraph(name="demo")
        graph.add_node("vision", VisionNode(prompt="What do you see?"))
        graph.add_node("reason", ReasoningNode(prompt="What should I do?"))
        graph.add_node("act", ActionNode(action_type="click"))
        graph.connect("vision", "reason")
        graph.connect("reason", "act")
        graph = graph.compile()

        final = graph.run(image="fake.png", prompt="login")

        assert final.step_count == 3
        assert [h.node_name for h in final.history] == [
            "VisionNode",
            "ReasoningNode",
            "ActionNode",
        ]
        assert len(final.actions_taken) == 1

    def test_decision_node_overrides_graph_edges(self):
        graph = VisionGraph()
        graph.add_node("check", DecisionNode(logic=lambda state: "act"))
        graph.add_node("wait", ActionNode(action_type="wait", name="wait"))
        graph.add_node("act", ActionNode(action_type="click", name="act"))
        # An unconditional edge to "wait" exists, but the DecisionNode's
        # own choice ("act") must win.
        graph.connect("check", "wait")
        graph = graph.compile()

        final = graph.run()

        assert final.actions_taken[0]["action_type"] == "click"

    def test_conditional_edges_route_based_on_state(self):
        graph = VisionGraph()
        graph.add_node("reason", ReasoningNode(prompt="p"))
        graph.add_node("confident_path", ActionNode(action_type="confident", name="confident_path"))
        graph.add_node("unsure_path", ActionNode(action_type="unsure", name="unsure_path"))
        graph.connect("reason", "confident_path", condition=lambda s: s.confidence > 0.8)
        graph.connect("reason", "unsure_path", condition=lambda s: s.confidence <= 0.8)
        graph = graph.compile()

        # Stub ReasoningNode always yields confidence 0.0 -> should take the unsure path
        result = graph.run()
        assert result.actions_taken[-1]["action_type"] == "unsure"

    def test_max_steps_stops_an_infinite_loop(self):
        graph = VisionGraph()
        graph.add_node("act", ActionNode(action_type="loop"))
        graph.connect("act", "act")  # self-loop, would run forever otherwise
        graph = graph.compile()

        final = graph.run(max_steps=5)

        assert final.step_count == 5
        assert len(final.actions_taken) == 5


# ---------------------------------------------------------------------------
# Conditional decision-making: Vision -> Decision -> (Action | Reasoning)
# ---------------------------------------------------------------------------


class TestConditionalRouting:
    """Vision -> Decision, branching to Action when a button is found and to
    Reasoning when it isn't. Verifies the graph actually takes the correct
    branch, not just that each node type works in isolation."""

    @staticmethod
    def _build_graph(button_present: bool) -> VisionGraph:
        def fake_infer(image, prompt):
            if button_present:
                return {
                    "objects": [{"type": "button", "label": "Submit", "confidence": 0.95}],
                    "scene": "form with a visible submit button",
                }
            return {"objects": [], "scene": "form with no button visible"}

        def decide_logic(state):
            found = any(obj.type == "button" for obj in state.visual_state.objects)
            return "action" if found else "reasoning"

        graph = VisionGraph(name="routing-test")
        graph.add_node("vision", VisionNode(prompt="Find the submit button", infer_fn=fake_infer))
        graph.add_node("decide", DecisionNode(logic=decide_logic))
        graph.add_node("action", ActionNode(action_type="click"))
        graph.add_node("reasoning", ReasoningNode(prompt="Why is the button missing?"))
        graph.connect("vision", "decide")
        return graph.compile()

    def test_button_found_branches_to_action(self):
        final = self._build_graph(button_present=True).run(image="screenshot.png")

        assert [h.node_name for h in final.history] == ["VisionNode", "DecisionNode", "ActionNode"]
        assert len(final.actions_taken) == 1
        assert final.actions_taken[0]["action_type"] == "click"
        # The Reasoning branch was not taken
        assert final.reasoning == ""

    def test_button_missing_branches_to_reasoning(self):
        final = self._build_graph(button_present=False).run(image="screenshot.png")

        assert [h.node_name for h in final.history] == ["VisionNode", "DecisionNode", "ReasoningNode"]
        assert final.reasoning == "[no reasoning backend configured]"
        # The Action branch was not taken
        assert final.actions_taken == []


# ---------------------------------------------------------------------------
# Verification loop: Vision -> Decision -> Action -> Vision -> Decision -> stop
#
# No VerificationNode class. "Verify the action worked" is just: re-run the
# same VisionNode, then let a DecisionNode inspect the new VisualState.
# This also doubles as the contract test for swapping infer_fn for a real
# VLM later: infer_fn(image, prompt) -> dict(objects=[...], scene=str, ...).
# ---------------------------------------------------------------------------


class TestVerificationLoop:
    def test_retries_action_then_stops_once_observation_shows_success(self):
        calls = {"n": 0}

        def fake_vlm_infer(image, prompt):
            """Stands in for a real VLM call (e.g. Phi-3.5-Vision). First
            observation shows a login error; every observation after the
            retry action shows success. Swapping this for a real model
            later should require no changes to VisionNode/DecisionNode/
            ActionNode or how the graph is wired."""
            calls["n"] += 1
            if calls["n"] == 1:
                return {
                    "objects": [{"type": "error", "label": "Invalid password", "confidence": 0.91}],
                    "scene": "login form showing an error",
                }
            return {
                "objects": [{"type": "success", "label": "Dashboard loaded", "confidence": 0.97}],
                "scene": "logged in successfully",
            }

        def decide_logic(state):
            has_error = any(obj.type == "error" for obj in state.visual_state.objects)
            return "action" if has_error else None  # None = no override, let the graph stop

        graph = VisionGraph(name="login-retry")
        graph.add_node("vision", VisionNode(prompt="What's the login state?", infer_fn=fake_vlm_infer))
        graph.add_node("decide", DecisionNode(logic=decide_logic))
        graph.add_node("action", ActionNode(action_type="click_retry"))
        graph.connect("vision", "decide")
        graph.connect("action", "vision")  # loop back: re-observe after acting
        graph = graph.compile()

        final = graph.run(prompt="What's the login state?", max_steps=10)

        assert calls["n"] == 2  # observed once before the retry, once after
        assert [h.node_name for h in final.history] == [
            "VisionNode",
            "DecisionNode",
            "ActionNode",
            "VisionNode",
            "DecisionNode",
        ]
        assert len(final.actions_taken) == 1  # retried exactly once, not looping forever
        assert final.visual_state.scene_description == "logged in successfully"
        assert final.step_count < 10  # stopped on its own, didn't hit max_steps
