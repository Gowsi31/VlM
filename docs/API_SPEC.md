# VisionGraph API Specification (v0.1)

## Core Concepts

### Agent
A visual agent that continuously observes, reasons, and acts on visual input.

```python
from visiongraph import VisionGraph

agent = VisionGraph(
    model="phi-3.5-vision",
    name="desktop-observer"
)
```

### Observation
Continuous or discrete visual input from cameras, screenshots, or video streams.

```python
agent.observe(camera_feed)  # Continuous stream
agent.observe(screenshot)   # Single image
agent.observe(video_path)   # Video file
```

### Nodes
Processing units in the agent's reasoning pipeline.

**Types:**
- `VisionNode` — Visual perception (what do I see?)
- `SpatialNode` — Spatial reasoning (where is it? how is it arranged?)
- `TemporalNode` — Temporal reasoning (how did it change?)
- `ReasoningNode` — LLM reasoning
- `DecisionNode` — Branch logic
- `ActionNode` — Execute action
- `MemoryNode` — Query/store memory
- `HumanNode` — Human approval

### State
Persistent state flowing through the agent lifecycle.

```python
class AgentState:
    # Visual
    current_frame: np.ndarray
    visual_observations: List[VisualObservation]
    
    # Spatial
    spatial_map: SpatialMap
    object_positions: Dict[str, BoundingBox]
    
    # Temporal
    frame_history: List[Frame]
    change_log: List[Change]
    temporal_context: Dict
    
    # Reasoning
    reasoning_chain: List[str]
    confidence: float
    
    # Memory
    working_memory: Dict
    long_term_memory: List[MemoryItem]
    
    # Action
    pending_actions: List[Action]
    executed_actions: List[ActionResult]
```

---

## API Reference

### VisionGraph Class

#### `__init__(model, name, config)`
Initialize a vision agent.

```python
agent = VisionGraph(
    model="phi-3.5-vision",
    name="screenshot-analyzer",
    config={
        "fps": 30,                    # Observation frequency
        "spatial_resolution": 1024,   # Spatial map resolution
        "memory_size": 1000,          # Observation history
        "max_reasoning_steps": 10
    }
)
```

#### `observe(input_source)`
Set visual input stream (camera, video, images, or function).

```python
# Camera stream (continuous)
agent.observe(camera_id=0)

# Video file
agent.observe("video.mp4")

# Screenshot function (called repeatedly)
agent.observe(lambda: pyautogui.screenshot())

# Image folder (read in sequence)
agent.observe("./screenshots/")

# Callback function
agent.observe(get_frame_function)
```

**Returns:** `ObservationStream` object

#### `add_node(name, node)`
Add a node to the agent's reasoning pipeline.

```python
agent.add_node("vision", VisionNode(
    prompt="What objects are visible?",
    output_format="json"
))

agent.add_node("spatial", SpatialNode(
    task="map_objects",  # or "track_movement", "detect_proximity"
    resolution=1024
))

agent.add_node("temporal", TemporalNode(
    task="detect_changes",
    memory_frames=30
))

agent.add_node("decide", DecisionNode(
    logic=lambda state: "act" if state.confidence > 0.8 else "observe"
))

agent.add_node("act", ActionNode(
    action_type="mouse_click",
    config={"coordinates_from": "spatial.detected_target"}
))
```

#### `connect(source, target, condition=None)`
Create edges in the reasoning pipeline.

```python
# Simple sequential
agent.connect("observe", "vision")
agent.connect("vision", "spatial")
agent.connect("spatial", "decide")
agent.connect("decide", "act")

# Conditional routing
agent.connect("decide", "memory",
    condition=lambda state: state.reasoning.startswith("Need"))

# Looping (continuous feedback)
agent.connect("act", "observe")  # After action, observe result
```

#### `compile(mode="continuous")`
Prepare agent for execution.

```python
# Continuous observation loop
agent = agent.compile(mode="continuous")

# Triggered on events
agent = agent.compile(mode="event", events=["frame_change"])

# Single run
agent = agent.compile(mode="single")
```

#### `run()`
Start the agent's reasoning and action loop.

```python
# Blocking (continuous operation)
agent.run()

# Non-blocking (in background)
agent.run(blocking=False)

# With timeout
agent.run(timeout_seconds=300)

# With callback
agent.run(on_frame=lambda state: print(state.visual_observations[-1]))
```

#### `stop()`
Stop the agent.

```python
agent.stop()
```

#### `get_state()`
Get current agent state.

```python
state = agent.get_state()
print(state.spatial_map)
print(state.reasoning_chain)
```

---

## Node Types

### VisionNode
Visual perception - what do I see?

```python
VisionNode(
    prompt="Describe the scene. List all objects.",
    output_format="json",
    vision_tools=["object_detection", "ocr", "scene_understanding"],
    cache_observations=True  # Cache duplicate frames
)
```

**Output:**
```json
{
  "objects": [
    {
      "type": "button",
      "label": "Submit",
      "confidence": 0.95,
      "description": "Blue button in center"
    }
  ],
  "scene": {
    "description": "Login form on white background",
    "lighting": "bright",
    "layout": "centered vertical"
  },
  "text": ["Username", "Password", "Forgot password?"],
  "timestamp": 1234567890.5
}
```

### SpatialNode
Spatial reasoning - where is it? how is it arranged?

```python
SpatialNode(
    task="map_objects",  # or "track", "detect_proximity", "path_planning"
    resolution=1024,
    include_depth=False  # For 3D spatial understanding
)
```

**Subtasks:**

#### `map_objects`
Build spatial map of detected objects.

```python
# Output: 2D spatial grid with object locations
spatial_state = {
    "objects_at": {
        "button_submit": {"x": 500, "y": 450, "w": 100, "h": 40},
        "text_input": {"x": 300, "y": 300, "w": 400, "h": 40}
    },
    "spatial_grid": [[0, 1, 0], [0, 2, 0], [0, 0, 0]],
    "relationships": [
        {"obj1": "text_input", "obj2": "button_submit", "relation": "below"},
        {"obj1": "label", "obj2": "text_input", "relation": "above"}
    ]
}
```

#### `track_movement`
Track how objects move over time.

```python
# Output: Movement vectors and trajectories
movement = {
    "objects": {
        "cursor": {"current": [500, 300], "previous": [480, 280], "velocity": [20, 20]},
        "window": {"moving": False}
    },
    "changes_detected": [
        {"object": "cursor", "type": "movement", "magnitude": 28.3, "direction": "southeast"}
    ]
}
```

#### `detect_proximity`
Detect proximity relationships between objects.

```python
# Output: Objects near each other
proximity = {
    "close_pairs": [
        {"obj1": "cursor", "obj2": "button_submit", "distance": 15, "distance_type": "pixels"}
    ],
    "clusters": [
        {"objects": ["text_input", "label", "button_submit"], "center": [400, 350]}
    ]
}
```

### TemporalNode
Temporal reasoning - how did it change?

```python
TemporalNode(
    task="detect_changes",  # or "predict_next", "anomaly_detect"
    memory_frames=30,       # Keep last 30 frames
    analysis_window="5s"
)
```

**Subtasks:**

#### `detect_changes`
Find what changed between frames.

```python
# Output: Frame-by-frame changes
changes = {
    "frame_deltas": [
        {
            "timestamp": 1234567890.5,
            "from_frame": 123,
            "to_frame": 124,
            "changes": [
                {"type": "object_appeared", "object": "error_message", "confidence": 0.98},
                {"type": "position_changed", "object": "cursor", "pixels": 45},
                {"type": "text_changed", "field": "status", "new_text": "Processing..."}
            ]
        }
    ],
    "significant_events": [
        {"type": "error_appeared", "timestamp": 1234567890.7, "importance": "high"}
    ]
}
```

#### `predict_next`
Predict next visual state.

```python
# Output: Predicted state
prediction = {
    "next_frame_prediction": "User will see loading spinner",
    "confidence": 0.85,
    "expected_objects": ["spinner", "progress_bar"],
    "estimated_time_to_stable_state": 3.5
}
```

#### `anomaly_detect`
Detect unusual visual patterns.

```python
# Output: Anomalies
anomalies = {
    "detected": [
        {"type": "unusual_layout", "severity": "medium", "description": "Elements shifted unexpectedly"},
        {"type": "performance_issue", "severity": "high", "description": "Frame rate dropped"}
    ]
}
```

### DecisionNode
Branch logic - what should I do?

```python
DecisionNode(
    logic=lambda state: (
        "investigate_error" if "error" in state.reasoning
        else "continue_observing"
    ),
    confidence_threshold=0.8
)
```

### ActionNode
Execute action.

```python
ActionNode(
    action_type="mouse_click",  # or "type", "scroll", "wait", "api_call"
    config={
        "target": "spatial.detected_button",
        "duration": 0.1
    }
)

# Or more complex
ActionNode(
    action_type="sequence",
    actions=[
        {"type": "click", "target": [500, 300]},
        {"type": "type", "text": "search query"},
        {"type": "click", "target": [550, 300]}
    ]
)
```

### MemoryNode
Query or store memories.

```python
MemoryNode(
    operation="store",  # or "recall", "search"
    memory_type="long_term",
    key="error_solutions",
    retention="persistent"
)

# Recall
MemoryNode(
    operation="recall",
    query="Solutions for 404 errors",
    count=5
)
```

### TemporalNode Example
Detect when something stabilizes.

```python
TemporalNode(
    task="detect_stability",
    stability_threshold=0.95,  # 95% similar frames
    required_frames=5          # 5 consecutive similar frames
)
```

---

## Complete Workflow Example

### Desktop GUI Agent

```python
from visiongraph import (
    VisionGraph, VisionNode, SpatialNode, TemporalNode,
    DecisionNode, ActionNode
)

# Create agent
agent = VisionGraph(
    model="phi-3.5-vision",
    name="desktop-assistant"
)

# Visual input (continuous screenshot)
agent.observe(lambda: pyautogui.screenshot())

# Perception pipeline
agent.add_node("vision", VisionNode(
    prompt="What's on screen? List all UI elements.",
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
    1. What's the user's current context?
    2. What happened since last observation?
    3. What should the agent do next?"""
))

agent.add_node("decide", DecisionNode(
    logic=lambda state: (
        "help_user" if any(
            "error" in str(obj).lower() 
            for obj in state.visual_observations[-1].objects
        ) else "monitor"
    )
))

agent.add_node("help", ActionNode(
    action_type="search",
    config={"query_from": "state.reasoning"}
))

agent.add_node("monitor", ActionNode(
    action_type="wait",
    config={"seconds": 5}
))

# Connect
agent.connect("observe", "vision")
agent.connect("vision", "spatial")
agent.connect("spatial", "temporal")
agent.connect("temporal", "analyze")
agent.connect("analyze", "decide")
agent.connect("decide", "help",
    condition=lambda s: s.next_decision == "help_user")
agent.connect("decide", "monitor",
    condition=lambda s: s.next_decision == "monitor")
agent.connect("help", "observe")      # Loop back
agent.connect("monitor", "observe")   # Loop back

# Compile and run
agent = agent.compile(mode="continuous")
agent.run()
```

---

## State Example

```python
state = agent.get_state()

# Visual observations
print(state.visual_observations[-1])
# {
#   "objects": [{"type": "button", "label": "Click me", ...}],
#   "scene": {"description": "Login form", ...}
# }

# Spatial map
print(state.spatial_map.get_object("button_submit"))
# {"x": 500, "y": 450, "w": 100, "h": 40}

# Temporal changes
print(state.change_log[-1])
# {"type": "object_appeared", "object": "error_message"}

# Reasoning
print(state.reasoning_chain[-1])
# "User is attempting to login but entered invalid password"

# Memory
print(state.working_memory.get("current_task"))
# "help user fix login error"
```

---

## Configuration

```python
agent = VisionGraph(
    model="phi-3.5-vision",
    config={
        # Observation
        "fps": 2,                    # Frames per second (2 = observe every 0.5s)
        "frame_skip": 1,             # Process every Nth frame
        
        # Spatial
        "spatial_resolution": 1024,  # Pixel resolution for spatial grid
        "spatial_grid_size": (10, 10),  # Cells for coarse spatial map
        
        # Temporal
        "memory_frames": 60,         # Keep last 60 observations
        "temporal_window": "10s",    # Analyze last 10 seconds
        
        # Reasoning
        "max_reasoning_steps": 20,
        "confidence_threshold": 0.8,
        "temperature": 0.7,
        
        # Caching
        "cache_vlm_responses": True,
        "cache_ttl": 5,              # Cache 5 seconds
        
        # Debugging
        "verbose": False,
        "save_frames": False,        # Save observation frames
        "save_dir": "./agent_logs/"
    }
)
```

---

## Error Handling

```python
from visiongraph.exceptions import (
    VisionNodeError,
    SpatialNodeError,
    TemporalNodeError,
    ActionNodeError,
    AgentError
)

try:
    agent.run()
except VisionNodeError as e:
    print(f"Vision processing failed: {e}")
except SpatialNodeError as e:
    print(f"Spatial reasoning failed: {e}")
except TemporalNodeError as e:
    print(f"Temporal analysis failed: {e}")
except ActionNodeError as e:
    print(f"Action execution failed: {e}")
except AgentError as e:
    print(f"Agent error: {e}")
```

---

## Advanced Features

### Streaming State
Get real-time state updates.

```python
for state_update in agent.stream_state():
    print(f"Frame {state_update.frame_count}")
    print(f"Objects: {len(state_update.spatial_map.objects)}")
    print(f"Changes: {state_update.change_log}")
```

### Callbacks
React to agent events.

```python
agent.on("frame_received", lambda frame: print("📸 Frame received"))
agent.on("object_detected", lambda obj: print(f"🔍 Detected: {obj.type}"))
agent.on("decision_made", lambda decision: print(f"🤔 Decision: {decision}"))
agent.on("action_taken", lambda action: print(f"⚡ Action: {action.type}"))
agent.on("error", lambda error: print(f"❌ Error: {error}"))
```

### Debugging
Inspect agent reasoning.

```python
# Get reasoning chain
agent.get_reasoning_chain()

# Get decision log
agent.get_decision_log()

# Save state snapshot
agent.save_state_snapshot("debug_state.json")

# Replay observations
agent.replay(start_frame=100, end_frame=200)
```

---

This is the new VisionGraph API. Ready to implement in Phase 2!
