# VisionGraph Examples

Real workflows using VisionGraph's observation + spatial + temporal capabilities.

## Example 1: Desktop GUI Agent (Continuous)

Continuous agent that monitors desktop for errors and helps user.

```python
from visiongraph import VisionGraph, VisionNode, SpatialNode, TemporalNode, DecisionNode, ActionNode
import pyautogui

# Create agent
agent = VisionGraph(
    model="phi-3.5-vision",
    name="desktop-helper",
    config={"fps": 2, "memory_frames": 30}
)

# Continuous screenshot observation
agent.observe(lambda: pyautogui.screenshot())

# Perception nodes
agent.add_node("vision", VisionNode(
    prompt="""Analyze the desktop screen:
    1. What application is active?
    2. What UI elements are visible?
    3. Are there any error messages?
    Return JSON.""",
    output_format="json"
))

agent.add_node("spatial", SpatialNode(
    task="map_objects",
    resolution=1024
))

agent.add_node("temporal", TemporalNode(
    task="detect_changes",
    memory_frames=30
))

# Reasoning
agent.add_node("analyze", ReasoningNode(
    prompt="""Based on vision + spatial + temporal analysis:
    1. What's happening on screen?
    2. Did anything change?
    3. Is there a problem that needs help?
    4. What should the agent do?"""
))

# Decision
agent.add_node("decide", DecisionNode(
    logic=lambda state: (
        "help_error" if any("error" in str(o).lower() for o in state.visual_observations[-1].objects)
        else "continue"
    )
))

# Actions
agent.add_node("help", ActionNode(
    action_type="search",
    config={"query_from": "state.reasoning", "open_browser": True}
))

agent.add_node("continue", ActionNode(
    action_type="wait",
    config={"seconds": 5}
))

# Orchestration
agent.connect("observe", "vision")
agent.connect("vision", "spatial")
agent.connect("spatial", "temporal")
agent.connect("temporal", "analyze")
agent.connect("analyze", "decide")
agent.connect("decide", "help",
    condition=lambda s: s.next_decision == "help_error")
agent.connect("decide", "continue",
    condition=lambda s: s.next_decision == "continue")
agent.connect("help", "observe")      # Loop back
agent.connect("continue", "observe")  # Loop back

# Run continuously
agent = agent.compile(mode="continuous")
agent.run()
```

---

## Example 2: Spatial Movement Tracking

Track object movements and detect patterns.

```python
from visiongraph import VisionGraph, VisionNode, SpatialNode, TemporalNode, ActionNode

agent = VisionGraph(model="phi-3.5-vision", name="movement-tracker")

# Video input
agent.observe("video.mp4")

agent.add_node("vision", VisionNode(
    prompt="Detect all moving objects",
    output_format="json"
))

agent.add_node("spatial", SpatialNode(
    task="track_movement"  # Track how objects move
))

agent.add_node("temporal", TemporalNode(
    task="detect_patterns",  # Find movement patterns
    memory_frames=60
))

agent.add_node("report", ActionNode(
    action_type="return_result",
    config={
        "format": "json",
        "include": ["trajectories", "speed", "patterns"]
    }
))

agent.connect("observe", "vision")
agent.connect("vision", "spatial")
agent.connect("spatial", "temporal")
agent.connect("temporal", "report")

agent = agent.compile(mode="single")
result = agent.run()

# Output
print(result)
# {
#   "objects": [
#     {
#       "id": "person_1",
#       "trajectory": [[100, 200], [150, 220], ...],
#       "speed": 45,  # pixels per second
#       "direction": "northeast",
#       "pattern": "pacing"
#     }
#   ]
# }
```

---

## Example 3: Temporal Change Detection (Video Analysis)

Analyze video for interesting changes.

```python
from visiongraph import VisionGraph, VisionNode, TemporalNode, ActionNode

agent = VisionGraph(model="phi-3.5-vision", name="change-detector")

agent.observe("interview.mp4")

agent.add_node("vision", VisionNode(
    prompt="What's happening in this frame?"
))

agent.add_node("temporal", TemporalNode(
    task="detect_changes",
    memory_frames=30,
    analysis_window="5s"
))

agent.add_node("annotate", ActionNode(
    action_type="generate_captions",
    config={"include_timestamps": True}
))

agent.connect("observe", "vision")
agent.connect("vision", "temporal")
agent.connect("temporal", "annotate")

agent = agent.compile(mode="single")
result = agent.run()

# Output: Timestamps of interesting changes
# 0:05 - Speaker's facial expression changed
# 0:15 - New person appeared on screen
# 0:42 - Light dimmed significantly
```

---

## Example 4: Proximity-Based Actions

React when objects get close to each other.

```python
from visiongraph import VisionGraph, VisionNode, SpatialNode, DecisionNode, ActionNode

agent = VisionGraph(model="phi-3.5-vision", name="proximity-detector")

# Camera feed
agent.observe(camera_id=0)

agent.add_node("vision", VisionNode(
    prompt="Detect all people and objects"
))

agent.add_node("spatial", SpatialNode(
    task="detect_proximity",  # Special task for proximity
    proximity_threshold=50  # pixels
))

agent.add_node("decide", DecisionNode(
    logic=lambda state: (
        "alert" if len(state.spatial_map.close_pairs) > 0
        else "monitor"
    )
))

agent.add_node("alert", ActionNode(
    action_type="send_notification",
    config={"message": "Objects are close"}
))

agent.add_node("monitor", ActionNode(
    action_type="wait",
    config={"seconds": 2}
))

agent.connect("observe", "vision")
agent.connect("vision", "spatial")
agent.connect("spatial", "decide")
agent.connect("decide", "alert")
agent.connect("decide", "monitor")
agent.connect("alert", "observe")
agent.connect("monitor", "observe")

agent = agent.compile(mode="continuous")
agent.run()
```

---

## Example 5: Stability Detection (Wait for Stable State)

Wait for a page to fully load by detecting when it stabilizes.

```python
from visiongraph import VisionGraph, VisionNode, TemporalNode, ActionNode

agent = VisionGraph(model="phi-3.5-vision", name="stability-waiter")

agent.observe(lambda: selenium_driver.get_screenshot())

agent.add_node("vision", VisionNode(
    prompt="Analyze page content"
))

agent.add_node("temporal", TemporalNode(
    task="detect_stability",
    stability_threshold=0.95,  # 95% similar
    required_frames=5          # 5 consecutive frames
))

agent.add_node("continue", ActionNode(
    action_type="proceed",
    config={"message": "Page is stable, proceeding..."}
))

agent.connect("observe", "vision")
agent.connect("vision", "temporal")
agent.connect("temporal", "continue")

agent = agent.compile(mode="continuous")
agent.run()
# Automatically stops when page stabilizes
```

---

## Example 6: UI Regression Testing with Spatial Layout

Compare layouts spatially and detect visual regressions.

```python
from visiongraph import VisionGraph, VisionNode, SpatialNode, TemporalNode, DecisionNode, ActionNode

agent = VisionGraph(model="phi-3.5-vision", name="ui-regression-tester")

# Current version
agent.observe("current_screenshot.png")

agent.add_node("vision", VisionNode(
    prompt="Extract all UI elements and their properties"
))

agent.add_node("spatial_current", SpatialNode(
    task="map_objects"
))

agent.add_node("load_baseline", ToolNode(
    tool_type="database",
    tool_config={"query": "SELECT spatial_map FROM ui_baselines WHERE test='login_form'"}
))

agent.add_node("compare", VisionNode(
    prompt="""Compare the spatial maps:
    1. Did element positions change?
    2. Are there layout differences?
    3. Are elements the right size?"""
))

agent.add_node("decide", DecisionNode(
    logic=lambda s: "report_issue" if s.reasoning.startswith("Changes") else "pass"
))

agent.add_node("report", ActionNode(
    action_type="generate_report",
    config={"format": "html", "include_diffs": True}
))

agent.connect("observe", "vision")
agent.connect("vision", "spatial_current")
agent.connect("spatial_current", "load_baseline")
agent.connect("load_baseline", "compare")
agent.connect("compare", "decide")
agent.connect("decide", "report")

agent = agent.compile(mode="single")
result = agent.run()
```

---

## Example 7: Real-time Gesture Recognition

Detect and respond to hand gestures.

```python
from visiongraph import VisionGraph, VisionNode, SpatialNode, TemporalNode, DecisionNode, ActionNode

agent = VisionGraph(model="phi-3.5-vision", name="gesture-recognizer")

agent.observe(camera_id=0)

agent.add_node("vision", VisionNode(
    prompt="Detect hands and finger positions"
))

agent.add_node("spatial", SpatialNode(
    task="track_movement",
    include_depth=True  # Track hand positions in space
))

agent.add_node("temporal", TemporalNode(
    task="detect_patterns",  # Recognize gestures from movement
    memory_frames=15         # Last 15 frames = ~0.5s at 30fps
))

agent.add_node("decide", DecisionNode(
    logic=lambda state: (
        "thumbs_up" if "thumbs up" in state.reasoning
        else "thumbs_down" if "thumbs down" in state.reasoning
        else "other"
    )
))

agent.add_node("respond_up", ActionNode(
    action_type="volume_up"
))

agent.add_node("respond_down", ActionNode(
    action_type="volume_down"
))

agent.add_node("other", ActionNode(
    action_type="wait"
))

agent.connect("observe", "vision")
agent.connect("vision", "spatial")
agent.connect("spatial", "temporal")
agent.connect("temporal", "decide")
agent.connect("decide", "respond_up", 
    condition=lambda s: s.next_decision == "thumbs_up")
agent.connect("decide", "respond_down",
    condition=lambda s: s.next_decision == "thumbs_down")
agent.connect("decide", "other")
agent.connect("respond_up", "observe")
agent.connect("respond_down", "observe")
agent.connect("other", "observe")

agent = agent.compile(mode="continuous")
agent.run()
```

---

## Example 8: State-Machine Behavior (Browser Testing)

Multi-step workflow with state machines.

```python
from visiongraph import VisionGraph, VisionNode, SpatialNode, DecisionNode, ActionNode

agent = VisionGraph(model="phi-3.5-vision", name="browser-tester")

agent.observe(lambda: selenium_driver.get_screenshot())

# State 1: Find login form
agent.add_node("find_form", VisionNode(
    prompt="Is there a login form?"
))

agent.add_node("form_spatial", SpatialNode(task="map_objects"))

agent.add_node("decide_form", DecisionNode(
    logic=lambda s: "fill_form" if "form" in s.reasoning else "find_form"
))

# State 2: Fill login
agent.add_node("fill_login", ActionNode(action_type="click", config={"target": "email_field"})
agent.add_node("type_email", ActionNode(action_type="type", config={"text": "user@test.com"})

# State 3: Submit
agent.add_node("click_submit", ActionNode(action_type="click", config={"target": "submit_button"})

# State 4: Verify success
agent.add_node("verify", VisionNode(
    prompt="Is the user now logged in?"
))

agent.add_node("report", ActionNode(
    action_type="return_result",
    config={"status": "from:verify"}
))

# State machine connections
agent.connect("observe", "find_form")
agent.connect("find_form", "form_spatial")
agent.connect("form_spatial", "decide_form")
agent.connect("decide_form", "fill_login",
    condition=lambda s: s.next_decision == "fill_form")
agent.connect("fill_login", "type_email")
agent.connect("type_email", "click_submit")
agent.connect("click_submit", "verify")
agent.connect("verify", "report")

agent = agent.compile(mode="single")
result = agent.run()
```

---

## Running Examples

All examples work in Colab:

```bash
# In Colab
!git clone https://github.com/YOU/visiongraph.git
%cd visiongraph
!pip install -e .

# Run example
%run examples/desktop_gui_agent.py
```

See [API_SPEC.md](./API_SPEC.md) for complete API documentation.
