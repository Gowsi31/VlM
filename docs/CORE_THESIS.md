# VisionGraph Core Thesis

## What is VisionGraph?

**VisionGraph is a vision-native workflow orchestration framework for building applications that perceive, understand, remember, and act on visual environments.**

**Tagline:** See → Understand → Act.

### What VisionGraph Is NOT

- ❌ A VLM wrapper
- ❌ "LangGraph but for vision"
- ❌ A no-code platform (this is code-first, framework-first)
- ❌ Designed to replace generic agent orchestrators
- ❌ A computer-use automation tool (though it can do that)

### What VisionGraph IS

- ✅ **Vision-native orchestration** — Designed from ground-up for visual perception workflows
- ✅ **Persistent visual state** — Scene understanding maintained across observations
- ✅ **Visual memory system** — Persistent representations of what the application has seen
- ✅ **Temporal reasoning** — Detect changes, track motion, understand sequences
- ✅ **Event detection** — Visual changes trigger workflow decisions
- ✅ **Developer SDK** — Framework for building visual applications (not just calling models)
- ✅ **Integrable** — Can work with/alongside LangGraph, agent frameworks, custom logic

---

## Architecture: The Perception-Reasoning-Action Loop

```
┌─────────────────────────────────────────────────────────────┐
│                   VisionGraph Agent                         │
│           (Persistent Visual Understanding)                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   PERCEPTION LAYER                                          │
│   ┌──────────────────────────────────────────────────────┐  │
│   │ Observation Input                                    │  │
│   │  • Image, Screenshot, Video Frame, Camera Stream    │  │
│   └────────────────┬─────────────────────────────────────┘  │
│                    │                                         │
│   ┌────────────────▼─────────────────────────────────────┐  │
│   │ Multimodal Perception                               │  │
│   │  • VLM (what is this?)                              │  │
│   │  • OCR (read text)                                  │  │
│   │  • Detector (find objects)                          │  │
│   │  • Tracker (follow entities)                        │  │
│   │  • Embeddings (semantic similarity)                 │  │
│   │  • Segmentation (boundaries)                        │  │
│   └────────────────┬─────────────────────────────────────┘  │
│                    │                                         │
│   UNDERSTANDING LAYER                                       │
│   ┌────────────────▼─────────────────────────────────────┐  │
│   │ Visual State (Persistent Scene Representation)       │  │
│   │                                                      │  │
│   │  Objects: {id, type, bbox, confidence, attrs}      │  │
│   │  Entities: person_1, person_2, object_3, ...       │  │
│   │  Spatial: left_of, right_of, near, inside, ...     │  │
│   │  Properties: color, size, text, state, ...         │  │
│   └────────────────┬─────────────────────────────────────┘  │
│                    │                                         │
│   MEMORY LAYER                                              │
│   ┌────────────────▼─────────────────────────────────────┐  │
│   │ Visual Memory                                        │  │
│   │                                                      │  │
│   │  Scene Memory: Current environment representation   │  │
│   │  Entity Memory: History of each object/person       │  │
│   │  Event Memory: Things that happened                 │  │
│   │  Semantic Memory: Important facts learned           │  │
│   │  Episodic Memory: Meaningful sequences              │  │
│   └────────────────┬─────────────────────────────────────┘  │
│                    │                                         │
│   EVENT DETECTION LAYER                                     │
│   ┌────────────────▼─────────────────────────────────────┐  │
│   │ Visual Events (Trigger Workflows)                    │  │
│   │                                                      │  │
│   │  • object_entered_area                              │  │
│   │  • object_disappeared                               │  │
│   │  • person_moved                                     │  │
│   │  • door_opened                                      │  │
│   │  • ui_changed                                       │  │
│   │  • unexpected_state                                 │  │
│   │  • custom_events                                    │  │
│   └────────────────┬─────────────────────────────────────┘  │
│                    │                                         │
│   REASONING LAYER                                           │
│   ┌────────────────▼─────────────────────────────────────┐  │
│   │ Agent Reasoning (Graph Execution)                   │  │
│   │                                                      │  │
│   │  Input: Current state + memory + events + user goal │  │
│   │  Process: LLM reasoning + graph logic               │  │
│   │  Output: Decision on what to do                     │  │
│   └────────────────┬─────────────────────────────────────┘  │
│                    │                                         │
│   ACTION LAYER                                              │
│   ┌────────────────▼─────────────────────────────────────┐  │
│   │ Actions (Executable by Agent)                       │  │
│   │                                                      │  │
│   │  • Tool calls (API, search, database)               │  │
│   │  • Computer interaction (click, type, scroll)       │  │
│   │  • Notifications (alert, log, report)               │  │
│   │  • Future: Robot actions, VLA control               │  │
│   └────────────────┬─────────────────────────────────────┘  │
│                    │                                         │
│   ┌────────────────▼─────────────────────────────────────┐  │
│   │ Observe Again (Feedback Loop)                       │  │
│   └────────────────┬─────────────────────────────────────┘  │
│                    │                                         │
│                    └──→ (Loop back to Observation)         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Core Concepts

### 1. Observation
**Everything starts with an observation.**

```python
# Input types
agent.observe(image)                    # Single image
agent.observe("screenshot.png")         # Screenshot file
agent.observe("video.mp4")              # Video frames
agent.observe(camera_id=0)              # Camera stream
agent.observe(lambda: pyautogui.screenshot())  # Function

# Output: Observation object
observation = {
    "timestamp": 1234567890.5,
    "source": "camera",
    "image": np.ndarray,
    "frame_id": 1234
}
```

Not just a prompt. An actual **perception event**.

---

### 2. Visual State
**Persistent representation of what exists in the scene.**

```python
class VisualState:
    # Objects currently visible
    objects: List[Object] = [
        {
            "id": "person_1",
            "type": "person",
            "bbox": [100, 200, 200, 500],
            "confidence": 0.98,
            "attributes": {
                "pose": "sitting",
                "facing": "camera",
                "expression": "focused"
            }
        },
        {
            "id": "button_submit",
            "type": "button",
            "bbox": [400, 600, 500, 650],
            "confidence": 0.95,
            "text": "Submit",
            "interactive": True
        }
    ]
    
    # Spatial relationships
    relationships: List[Relationship] = [
        {"obj1": "person_1", "obj2": "button_submit", "relation": "facing"},
        {"obj1": "button_submit", "obj2": "text_input", "relation": "right_of"}
    ]
    
    # Confidence in the state
    overall_confidence: float = 0.93
    
    # When this state was observed
    timestamp: float = 1234567890.5
    
    # What changed since last observation
    changes: List[Change] = [
        {"type": "appeared", "object": "button_submit"},
        {"type": "moved", "object": "person_1", "pixels": 45}
    ]
```

This is **not a prompt**. It's a **structured scene representation** maintained across time.

---

### 3. Visual Memory
**Multiple types of memory for different needs.**

```python
class VisualMemory:
    # Scene Memory: Current environment
    scene_memory: Dict = {
        "location": "office",
        "lighting": "bright",
        "layout": "desk + chair",
        "active_applications": ["browser", "terminal"],
        "recent_changes": [...]
    }
    
    # Entity Memory: History of objects
    entity_memory: Dict = {
        "person_1": {
            "first_seen": 1234567890.0,
            "appearances": 45,
            "locations_visited": [[100, 200], [150, 250], ...],
            "interactions": ["looked at screen", "typed", "clicked button"],
            "attributes_observed": {"color": "blue shirt", "pose": ["sitting", "standing"]}
        },
        "button_submit": {
            "first_seen": 1234567895.0,
            "clicked": True,
            "hover_count": 3
        }
    }
    
    # Event Memory: Things that happened
    event_memory: List[Event] = [
        {"type": "object_appeared", "object": "error_message", "timestamp": 1234567920.0},
        {"type": "person_moved", "object": "person_1", "distance": 50, "timestamp": 1234567925.0},
        {"type": "click", "object": "button_submit", "timestamp": 1234567930.0}
    ]
    
    # Semantic Memory: Important facts learned
    semantic_memory: Dict = {
        "user_intent": "trying to login",
        "current_task": "enter credentials",
        "obstacles": ["password field showing error"],
        "learned_patterns": [
            "user clicks submit after typing email",
            "errors appear in red text at bottom"
        ]
    }
    
    # Episodic Memory: Meaningful sequences
    episodic_memory: List[Episode] = [
        {
            "id": "login_attempt_1",
            "start": 1234567890.0,
            "end": 1234567935.0,
            "events": [...],
            "outcome": "error - invalid password",
            "key_frames": [frame_0, frame_10, frame_20]
        }
    ]
```

Each type serves a different purpose in agent reasoning.

---

### 4. Entity Tracking
**Same entity maintains identity across frames.**

```python
# Frame 1
person_1 = {
    "bbox": [100, 200, 180, 500],
    "pose": "sitting",
    "location": "left_side"
}

# Frame 2 (person moved slightly)
person_1 = {
    "bbox": [105, 205, 185, 505],  # Moved ~5 pixels
    "pose": "sitting",
    "location": "left_side"
}

# Frame 3 (person moved more)
person_1 = {
    "bbox": [150, 250, 230, 550],  # Moved ~45 pixels
    "pose": "standing",
    "location": "center"
}

# NOT treated as: person_1, person_2, person_3
# TRACKED as: person_1 across all frames
# Movement vector: [150-100, 250-200] = [50, 50] pixels
```

**Don't re-identify the same object as new every frame.**

---

### 5. Visual Events
**Events trigger agent decisions and workflows.**

```python
# Visual Events (automatically detected)
events = [
    {"type": "object_appeared", "object_id": "error_popup", "confidence": 0.97},
    {"type": "object_disappeared", "object_id": "loading_spinner", "confidence": 0.99},
    {"type": "person_moved", "object_id": "person_1", "distance": 50, "direction": "right"},
    {"type": "property_changed", "object_id": "text_input", "property": "value", "new_value": "user@example.com"},
    {"type": "state_changed", "object_id": "button_submit", "old_state": "disabled", "new_state": "enabled"},
    {"type": "interaction_detected", "type": "click", "object_id": "button_submit"},
    {"type": "unexpected_visual_state", "reason": "red error text appeared"}
]

# Events can trigger workflows
def on_event(event):
    if event.type == "object_appeared" and "error" in event.object_id:
        agent.trigger_workflow("help_with_error")
    elif event.type == "object_disappeared" and event.object_id == "loading_spinner":
        agent.trigger_workflow("verify_page_loaded")
```

---

### 6. Temporal Reasoning
**Compare observations, understand change over time.**

```python
# What changed between frame N and N+1?
temporal_analysis = {
    "frame_delta": {
        "new_objects": ["error_message"],
        "disappeared_objects": [],
        "moved_objects": [
            {"id": "cursor", "from": [500, 300], "to": [510, 305], "pixels": 11}
        ],
        "property_changes": [
            {"object": "text_input", "property": "value", "from": "", "to": "user@"}
        ]
    },
    
    # Sequence analysis
    "sequence": {
        "user_typed": "user@",
        "then_clicked": "button_submit",
        "then_error_appeared": True,
        "pattern_name": "failed_login_attempt"
    },
    
    # Predictions
    "predictions": {
        "next_likely_action": "user will correct password",
        "next_likely_visual": "password field highlighted",
        "confidence": 0.82
    }
}
```

---

### 7. Multimodal Perception
**Don't rely on VLM alone. Combine multiple models.**

```python
# Single image fed to multiple perception models
image = observation.image

# Model 1: VLM (what is happening?)
vlm_output = vlm_model(image, "Describe the scene")
# → "User is typing in a login form. There's an error message."

# Model 2: OCR (read text)
ocr_output = ocr_model(image)
# → {"text": ["Username", "Password", "Invalid credentials"], "locations": [...]}

# Model 3: Object Detector (find objects)
detector_output = yolo_model(image)
# → [{"bbox": [100, 200, 300, 250], "class": "person", "conf": 0.95}, ...]

# Model 4: Tracker (follow objects)
tracker_output = tracker.update(image)
# → {"person_1": [105, 205, ...], "button_submit": [400, 600, ...]}

# Model 5: Embeddings (semantic similarity)
embeddings = embedding_model(image)
# → [0.23, 0.45, 0.67, ...]  (compare with previous frames)

# Model 6: Segmentation (boundaries)
segmentation = segmentation_model(image)
# → mask showing regions (UI vs background vs people)

# Combine all into Visual State
visual_state = merge_perception_outputs(
    vlm_output,
    ocr_output,
    detector_output,
    tracker_output,
    embeddings,
    segmentation
)
```

The VLM is **one component**, not the only component.

---

### 8. Streaming
**From images → video → camera streams.**

**V1:** Single image
```python
agent.observe("screenshot.png")
result = agent.run()
```

**V2:** Video sequence
```python
agent.observe("video.mp4")
# Processes frame by frame
# Maintains state across frames
result = agent.run()
```

**V3:** Continuous camera stream
```python
agent.observe(camera_id=0)
# Real-time observation loop
# Immediate perception → reasoning → action
agent.run()  # Continuous
```

---

### 9. First Killer Demo: Desktop UI Automation

Desktop/UI automation is an ideal first demonstration of VisionGraph's abstractions:

```python
# Workflow: Automate a login sequence
workflow = VisionGraph(name="login-automation")

workflow.add_node("observe", ObservationNode(
    source=lambda: pyautogui.screenshot()
))

workflow.add_node("perceive", VisionNode(
    prompt="Find the login form. What fields are visible?"
))

workflow.add_node("reason", ReasoningNode(
    prompt="Should we fill the form? What's the next step?"
))

workflow.add_node("act", ActionNode(
    actions=["click", "type", "submit"]
))

workflow.add_node("verify", VisionNode(
    prompt="Did the login succeed or is there an error?"
))

# Connect
workflow.connect("observe", "perceive")
workflow.connect("perceive", "reason")
workflow.connect("reason", "act")
workflow.connect("act", "observe")

# Run the workflow
workflow = workflow.compile()
workflow.run()
```

**Why desktop automation is a good first demo:**
- Clear success criteria (login succeeds or fails)
- Proves the observation → perception → action loop works
- Demonstrates value vs doing it with a generic agent framework
- Practical and immediately useful
- Not the only use case (also: visual QA, monitoring, inspection)

**But this is NOT VisionGraph's whole identity.** It's just the first vertical slice to prove the concepts.

---

### 10. Evaluation & Observability
**Measure performance, debug behavior.**

```python
# Evaluation metrics
metrics = {
    "perception_accuracy": 0.94,      # VLM correctness
    "tracking_accuracy": 0.91,        # Entity tracking correctness
    "event_detection": 0.89,          # Visual events detected correctly
    "memory_retrieval": 0.87,         # Relevant memories recalled
    "reasoning_accuracy": 0.92,       # Correct decisions made
    "action_success": 0.88,           # Actions achieved goals
    "latency_ms": 1200,               # Time per observation cycle
    "memory_usage_mb": 450
}

# Observability (what's the agent doing?)
observable_state = {
    "current_observation": {...},
    "visual_state": {...},
    "detected_events": [...],
    "memory_accessed": [...],
    "reasoning_chain": "User clicked button → error appeared → user should retry",
    "model_calls": [
        {"model": "phi-3.5-vision", "latency_ms": 800, "tokens": 120},
        {"model": "yolo", "latency_ms": 150, "boxes": 5}
    ],
    "tool_calls": [{"tool": "search", "query": "fix login error", "results": 3}],
    "actions_taken": [{"type": "click", "target": "retry_button", "success": True}],
    "errors": []
}
```

Without these, you just have "a cool demo."

---

## Developer API (Framework-First)

VisionGraph is **code-first, framework-first**. You build visual workflows programmatically.

The API should feel like a workflow orchestration framework (similar to LangGraph, Airflow, Temporal).

```python
from visiongraph import VisionGraph, VisionNode, MemoryNode, EventNode, ActionNode

# Define a visual workflow
workflow = VisionGraph(name="ui-automation")

# Step 1: Observe the screen
workflow.add_node("observe", ObservationNode(
    source=lambda: pyautogui.screenshot()
))

# Step 2: Perceive what you see
workflow.add_node("perceive", VisionNode(
    prompt="What UI elements are visible?",
    models=["vlm", "ocr"]
))

# Step 3: Remember what you've learned
workflow.add_node("memory", MemoryNode(
    memory_types=["scene", "entity", "event"]
))

# Step 4: Detect visual events
workflow.add_node("events", EventNode(
    detect=["object_appeared", "object_moved", "state_changed"]
))

# Step 5: Reason about what to do
workflow.add_node("reason", ReasoningNode(
    prompt="Based on what you see, what should we do?"
))

# Step 6: Take action
workflow.add_node("act", ActionNode(
    actions=["click", "type", "scroll", "wait"]
))

# Step 7: Verify result
workflow.add_node("verify", VisionNode(
    prompt="Did the action succeed?"
))

# Wire up the workflow
workflow.connect("observe", "perceive")
workflow.connect("perceive", "memory")
workflow.connect("memory", "events")
workflow.connect("events", "reason")
workflow.connect("reason", "act")
workflow.connect("act", "observe")  # Loop back for continuous workflow

# Compile and run
workflow = workflow.compile(mode="continuous")
workflow.run()
```

This is **not** a no-code platform. It's a **developer framework** for building visual workflows.

---

## What NOT to Claim

❌ "We can send images to an LLM."  
❌ "We support GPT/Qwen/LLaVA."  
❌ "We have VisionNodes."  
❌ "We can call tools after looking at an image."  
❌ "We have an agent that understands screenshots."  
❌ "We use graphs for VLM workflows."  

Those are table stakes, not differentiation.

---

## What Makes VisionGraph Different

Vision-native workflow orchestration requires primitives that generic agent frameworks don't provide:

✅ **Visual State as first-class citizen** — Scene representation (objects, positions, relationships) not as a string but as structured data  
✅ **Visual Memory** — Persistent scene, entity, and event understanding across observations  
✅ **Entity tracking** — Maintaining identity of objects across frames (not re-identifying as new)  
✅ **Temporal reasoning** — Detecting changes, tracking motion, understanding sequences over time  
✅ **Visual Events** — "object appeared", "moved", "state changed" → trigger workflow logic  
✅ **Multimodal perception** — Combining VLM + OCR + detection + tracking → richer understanding  
✅ **Observation as loop** — Screenshot → perceive → remember → decide → act → screenshot → verify  

Generic agent frameworks optimize for LLM reasoning. VisionGraph optimizes for visual perception.

You can **use VisionGraph with LangGraph** — VisionGraph handles the visual workflow layer, LangGraph handles higher-level reasoning. They complement rather than compete.

---

## What V0.1 Should Prove

V0.1 is not "full visual agent platform." It's a minimal core that proves the thesis.

**V0.1 must show:**
1. Observation → Perception → Visual State works
2. Entity tracking maintains identity across frames  
3. Graph orchestration executes visual workflows
4. Visual State is useful (better than string-based)
5. One complete demo (desktop login) works end-to-end

**V0.1 scope:** Screenshot input, VLM perception, basic state, graph orchestration, basic actions, one demo.

**What V0.1 proves:** Visual-native orchestration is a different (and better) way to build perception workflows than generic agent orchestrators.

---

## Roadmap

### V0.1 (Weeks 2-5)
- Observation system (screenshots)
- Visual State (basic scene representation)
- Entity tracking (basic)
- Graph orchestration
- One demo: Desktop UI login automation
- Prove: Perception loop works

### V0.2 (Weeks 6-10)
- Visual Memory (scene + entity memory)
- Event detection ("object appeared", "moved", etc.)
- Temporal reasoning (detect changes)
- UI improvements
- Prove: Memory & events help workflows

### V0.3+ (Future)
- Video input, continuous streams
- Advanced spatial reasoning
- Multimodal perception
- Integration with other frameworks

---

## The Thesis, Core Points

**1. What is VisionGraph?**
Vision-native workflow orchestration framework. See → Understand → Act.

**2. What problem does it solve?**
Building perception-based workflows is hard with generic agent frameworks. VisionGraph is designed from the ground-up for visual tasks.

**3. Why does vision require specialized orchestration?**
Vision is stateful (entity tracking, scene memory, temporal change). Generic LLM orchestrators treat each input as independent. Visual workflows need persistent state.

**4. What are its core abstractions?**
Observation, Perception, Visual State, Memory, Events, Graph, Actions.

**5. How is it different from generic agent orchestration?**
Generic frameworks optimize for LLM reasoning. VisionGraph optimizes for visual perception. Visual State is first-class, not an afterthought.

**6. What will V0.1 prove?**
That visual-native orchestration enables simpler, more reliable visual workflows than generic agent frameworks.

---

See [PLANNING.md](./PLANNING.md) for implementation roadmap.
See [ARCHITECTURE.md](./ARCHITECTURE.md) for technical design.
See [docs/API_SPEC.md](./API_SPEC.md) for developer API.
