# VisionGraph Architecture Update - Spatial & Temporal Reasoning

## What Changed

The VisionGraph architecture has been upgraded to include **spatial and temporal reasoning** for building truly intelligent visual agents.

### From: Single-Frame Analysis
```python
# Before
graph = VisionGraph()
graph.addNode("analyze", VisionNode())
agent = graph.compile()
result = agent.run(image=screenshot)  # One-off analysis
```

### To: Continuous Visual Intelligence
```python
# After
agent = VisionGraph()
agent.observe(camera_feed)  # Continuous stream

agent.add_node("vision", VisionNode())
agent.add_node("spatial", SpatialNode(task="map_objects"))
agent.add_node("temporal", TemporalNode(task="detect_changes"))
agent.add_node("reason", ReasoningNode())
agent.add_node("decide", DecisionNode())
agent.add_node("act", ActionNode())

agent.connect("observe", "vision")...
agent.run()  # Continuous operation
```

---

## New Capabilities

### 1. Continuous Observation (`observe()`)
- Camera streams
- Video files
- Screenshot functions
- Image sequences

```python
agent.observe(camera_id=0)           # Webcam
agent.observe("video.mp4")           # Video file
agent.observe(lambda: pyautogui.screenshot())  # Screenshots
```

### 2. Spatial Reasoning (SpatialNode)
- **map_objects** — Build spatial map of scene
- **track_movement** — Track how objects move
- **detect_proximity** — Find nearby objects
- **path_planning** — Plan paths between objects

```python
agent.add_node("spatial", SpatialNode(
    task="map_objects",
    resolution=1024
))

# Output: Object positions, relationships, layouts
```

### 3. Temporal Reasoning (TemporalNode)
- **detect_changes** — What changed between frames?
- **predict_next** — What will happen next?
- **anomaly_detect** — Is this unusual?
- **detect_stability** — Wait for things to stabilize

```python
agent.add_node("temporal", TemporalNode(
    task="detect_changes",
    memory_frames=30
))

# Output: Change log, significant events, predictions
```

### 4. Persistent State
- Visual observations (frame history)
- Spatial map (object positions)
- Change log (temporal events)
- Working memory (current episode)
- Long-term memory (across episodes)

```python
state = agent.get_state()
print(state.spatial_map.objects_at)    # Current positions
print(state.change_log)                 # What changed
print(state.reasoning_chain)            # What was reasoned
print(state.memory.long_term)           # Learned facts
```

### 5. Real-time Agent Loop
- Continuous observation → Perception → Spatial → Temporal → Reasoning → Decision → Action → Loop back

```
[Observe] → [Vision] → [Spatial] → [Temporal] → [Reason] → [Decide] → [Act]
   ↑                                                                      ↓
   ←──────────────────── Loop Back ──────────────────────────────────────┘
```

---

## Updated Files

### Documentation (✅ Updated)
- **README.md** — New examples with spatial/temporal
- **ARCHITECTURE.md** — Complete spatial+temporal system architecture
- **project.md** — Updated vision and core concepts
- **docs/API_SPEC.md** — Complete new API with `observe()`, `SpatialNode`, `TemporalNode`
- **docs/EXAMPLES.md** — 8 real workflows showing spatial/temporal features

### Code Structure (Ready for Phase 2)
```
visiongraph/
├── graph.py              # Updated for observe(), continuous mode
├── nodes.py              # Added SpatialNode, TemporalNode
├── state.py              # Added spatial_map, change_log, frame_history
├── observation.py        # [NEW] Observation stream management
├── spatial.py            # [NEW] Spatial reasoning engine
├── temporal.py           # [NEW] Temporal reasoning engine
├── executor.py           # Updated for continuous loop
└── memory.py             # Working + long-term memory
```

---

## New Nodes

| Node Type | Task | Output |
|-----------|------|--------|
| **VisionNode** | What do I see? | Detected objects, scene, text |
| **SpatialNode** | Where is it? | Object positions, relationships |
| **TemporalNode** | What changed? | Change log, predictions, anomalies |
| **ReasoningNode** | What does it mean? | Reasoning chain, confidence |
| **DecisionNode** | What to do? | Next action decision |
| **ActionNode** | Do it | Execute click, type, search, etc. |
| **MemoryNode** | Remember? | Store/recall memories |
| **HumanNode** | Approve? | Pause for human input |

---

## New API Methods

```python
agent = VisionGraph(model="phi-3.5-vision")

# Observation
agent.observe(camera_feed)                    # Set input source
agent.get_frame()                             # Get current frame
agent.get_history(n_frames)                   # Get frame history

# Node management
agent.add_node(name, node)
agent.connect(source, target, condition)

# Compilation
agent.compile(mode="continuous"|"event"|"single")
agent.validate()                              # Check graph validity
agent.visualize()                             # Return mermaid diagram

# Execution
agent.run(timeout_seconds=300, blocking=True)
agent.stop()
agent.stream_state()                          # Real-time state updates

# State inspection
agent.get_state()
agent.get_reasoning_chain()
agent.get_decision_log()
agent.save_state_snapshot(filepath)

# Debugging
agent.on(event, callback)                     # Register event callbacks
agent.replay(start_frame, end_frame)          # Replay observations
```

---

## Example Workflows Added

1. **Desktop GUI Agent** — Continuous monitoring for errors
2. **Movement Tracking** — Track object motion patterns
3. **Video Change Detection** — Find important changes in video
4. **Proximity Detection** — React when objects get close
5. **Stability Detection** — Wait for page to load
6. **UI Regression Testing** — Spatial layout comparison
7. **Gesture Recognition** — Real-time hand gesture detection
8. **Browser Testing** — Multi-step state machine workflows

All ready to run in Google Colab.

---

## Configuration Options

```python
config = {
    # Observation
    "fps": 2,                          # Frames per second
    "frame_skip": 1,                   # Process every Nth frame
    
    # Spatial
    "spatial_resolution": 1024,        # Grid resolution
    "spatial_grid_size": (10, 10),    # Cells for coarse map
    
    # Temporal
    "memory_frames": 60,               # Frame history size
    "temporal_window": "10s",          # Analysis window
    
    # Reasoning
    "max_reasoning_steps": 20,
    "confidence_threshold": 0.8,
    
    # Caching
    "cache_vlm_responses": True,
    "cache_ttl": 5,
    
    # Debugging
    "verbose": False,
    "save_frames": False,
}

agent = VisionGraph(config=config)
```

---

## Phase 2: Implementation Priority

When coding, prioritize:

1. **Observation System** — Frame streams, queuing
2. **Graph Engine** — Core orchestration (existing)
3. **State Management** — Spatial + temporal state
4. **Spatial Node** — Object mapping, relationships
5. **Temporal Node** — Change detection, predictions
6. **Continuous Loop** — Agent execution cycle

---

## Backward Compatibility

✅ Old single-frame API still works:
```python
agent = VisionGraph()
agent.add_node("analyze", VisionNode())
agent.run(image=screenshot)  # Single run still supported
```

✅ New continuous API is additive:
```python
agent.observe(camera)  # New
agent.run()            # Continuous
```

---

## Ready for Phase 2

All architecture, API, and examples are now updated. You can start implementing the core framework in Google Colab with full clarity on:

- What nodes to build (Vision, Spatial, Temporal, Reasoning, Decision, Action)
- How state flows through the system
- What the persistence model is
- How continuous loops work
- Real-world examples to implement against

**Push to GitHub and start coding!**

See [API_SPEC.md](./docs/API_SPEC.md) for complete reference.
See [EXAMPLES.md](./docs/EXAMPLES.md) for 8 real workflows.
See [ARCHITECTURE.md](./docs/ARCHITECTURE.md) for system design.
