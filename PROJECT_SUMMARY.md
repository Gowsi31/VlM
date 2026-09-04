# VisionGraph — Complete Project Summary

## What We're Building

**VisionGraph is a vision-native workflow orchestration framework for building applications that perceive, understand, remember, and act on visual environments.**

**Tagline:** See → Understand → Act.

---

## The Vision (Long-term)

```
                    VisionGraph
                        │
    ┌───────────────────┼───────────────────┐
    ↓                   ↓                   ↓
Desktop UI         Physical World       Industrial
Automation         (Robotics)           (Inspection)
    │                   │                   │
    ↓                   ↓                   ↓
Screenshot          Camera              Factory
Browser             Robot               Quality
Mobile Device       Environment         Monitoring
```

**VisionGraph eventually enables:**
- Computer-use agents (desktop, browser, mobile)
- Robotic vision applications
- Industrial inspection and monitoring
- Any domain requiring visual perception → reasoning → action

But we're not building all of this now.

---

## The Approach: Proof → Expand

Instead of building the entire framework immediately, we're **proving the foundation, then expanding toward the vision.**

```
Your Original Vision (Full VisionGraph)
        │
        │
        ▼
┌───────────────────────────────────┐
│ Phase 0: V0.1 Technical Proof     │
│                                   │
│ Observation → Perception →        │
│ Visual State → Reasoning →        │
│ Action → Verification            │
│                                   │
│ Goal: Prove the abstraction works │
└───────────┬───────────────────────┘
            │
         SUCCESS?
            │
            ▼
┌───────────────────────────────────┐
│ Phase 1: V0.2 Visual State +      │
│ Memory + Events                   │
│                                   │
│ Add richer state representation   │
│ Add memory (scene, entity, event) │
│ Add visual event detection        │
└───────────┬───────────────────────┘
            │
            ▼
┌───────────────────────────────────┐
│ Phase 2: V0.3 Temporal +          │
│ Spatial Understanding             │
│                                   │
│ Add change detection              │
│ Add spatial relationships         │
│ Add pattern recognition           │
└───────────┬───────────────────────┘
            │
            ▼
┌───────────────────────────────────┐
│ Phase 3: V0.4 Specialized Nodes   │
│                                   │
│ ObserveNode, PerceptionNode,      │
│ SpatialNode, TemporalNode,        │
│ MemoryNode, EventNode, etc.       │
└───────────┬───────────────────────┘
            │
            ▼
┌───────────────────────────────────┐
│ Phase 4: V1.0 Full Framework      │
│                                   │
│ Vision-native orchestration       │
│ Complete developer SDK            │
│ Ready for production use          │
└───────────────────────────────────┘
```

---

## Phase 0: V0.1 Technical Proof (Weeks 1-8)

### Goal
Prove that this abstraction is better than `VLM → parse response → act`

### Build Only
```
Observation (screenshot)
     ↓
Perception (VLM: what do you see?)
     ↓
Visual State (structured objects, positions)
     ↓
Reasoning (LLM: what should we do?)
     ↓
Action (click, type, scroll)
     ↓
Verification (screenshot: did it work?)
     ↓
Loop
```

### Scope (What's IN)
- ✅ Observation system (static images, screenshots)
- ✅ VLM perception (what objects are visible?)
- ✅ Visual State (basic scene representation)
- ✅ Graph orchestration (connect nodes)
- ✅ Action execution (click, type, scroll, wait)
- ✅ Verification (re-observe and check)
- ✅ One complete demo (desktop login automation)
- ✅ Tests and documentation

### Scope (What's OUT)
- ❌ Entity tracking
- ❌ Video pipeline
- ❌ Visual memory
- ❌ Event detection
- ❌ Temporal reasoning
- ❌ Multi-model support
- ❌ No-code UI
- ❌ Production platform

### Success Criteria
**One question answered:** "Can developers express visual workflows more naturally in VisionGraph than stacking LangGraph + VLM?"

If YES → continue to V0.2  
If NO → ship it anyway as reference, learn and pivot

### Timeline
- Weeks 1-2: Observation + VLM client
- Weeks 2-3: Visual State (basic)
- Weeks 3-4: Graph orchestration
- Weeks 4-5: Action execution
- Weeks 5-6: Verification
- Weeks 6-7: One complete demo
- Weeks 7-8: Tests, docs, polish

---

## Phase 1: V0.2 Visual Intelligence (Weeks 9-20)

### Goal
Prove that Visual State + Memory + Events are valuable

### Add
- **Richer Visual State**
  ```python
  VisualState(
      scene=SceneDescription,
      objects=List[DetectedObject],
      relationships=List[Relationship],  # left_of, right_of, near, etc.
      spatial_context=SpatialMap,
      changes=ChangeLog,  # what changed since last observation
      confidence=float
  )
  ```

- **Visual Memory** (3 types, not 5)
  ```python
  SceneMemory      # Current environment state
  EntityMemory     # History of objects
  EventMemory      # Things that happened
  ```

- **Visual Event Detection**
  ```python
  object_appeared
  object_disappeared
  object_moved
  property_changed
  state_changed
  unexpected_state
  ```

- **Event-Triggered Workflows**
  ```python
  graph.on("object_appeared", handle_object)
  graph.on("state_changed", handle_change)
  ```

### Demo
- Visual workflow that uses memory to reason better
- Example: "Remember login failed before. Try different approach."

---

## Phase 2: V0.3 Temporal + Spatial (Weeks 21-32)

### Goal
Prove that temporal and spatial reasoning unlock capabilities impossible in V0.1/V0.2

### Add
- **Temporal Reasoning**
  - Detect changes between observations
  - Track sequences over time
  - Predict likely next state
  - Anomaly detection

- **Spatial Understanding**
  - Spatial relationships (above, below, left_of, contains)
  - Path planning
  - Proximity detection
  - Scene layout analysis

- **Temporal Events**
  - Pattern detected
  - Sequence recognized
  - Anomaly found
  - Stability reached

### Demo
- Workflow that tracks motion over time
- Workflow that detects unexpected visual patterns
- Workflow that reasons about spatial relationships

---

## Phase 3: V0.4 Specialized Nodes (Weeks 33-40)

### Goal
Build developer-friendly API with specialized vision nodes

### Add
```python
ObservationNode          # Handle image/screenshot/video input
PerceptionNode           # VLM-based perception
VisualStateNode          # Maintain and update state
SpatialNode              # Spatial reasoning
TemporalNode             # Temporal reasoning
MemoryNode               # Store/recall visual facts
EventDetectionNode       # Detect visual events
ReasoningNode            # LLM reasoning over state
ToolNode                 # Call external APIs
ActionNode               # Execute click/type/etc.
VerificationNode         # Verify action succeeded
LoopNode                 # Handle continuous loops
```

### Developer API
```python
graph = VisionGraph()

graph.observe(source)
graph.perceive(prompt)
graph.track()
graph.remember()
graph.detect_change()
graph.reason(prompt)
graph.act()
graph.verify()
graph.loop()
```

### Demo
- 5+ example workflows using specialized nodes
- Visual automation library
- Reusable patterns

---

## Phase 4: V1.0 Full Framework (Weeks 41-52)

### Goal
Mature, documented, production-ready framework

### Complete
- ✅ Full API documentation
- ✅ 10+ example workflows
- ✅ >80% test coverage
- ✅ Performance optimized
- ✅ Multiple observation sources (screenshot, video, camera, RTSP)
- ✅ Model adapters (VLM providers)
- ✅ Tool ecosystem
- ✅ Community contributions

### Expand
- Desktop automation
- Robotics/VLA
- Industrial inspection
- Any visual domain

---

## What Makes VisionGraph Different

**NOT:**
- ❌ "VLM wrapper"
- ❌ "LangGraph for vision"
- ❌ "Call a VLM and parse the response"

**IS:**
- ✅ **Visual State as first-class citizen** — Scene representation is structured data, not strings
- ✅ **Observation loops** — Screenshot → perceive → state → reason → act → screenshot → verify
- ✅ **Visual Memory** — Persistent understanding of scene, entities, events
- ✅ **Temporal reasoning** — Detect change, track motion, understand sequences
- ✅ **Visual Events** — "object appeared", "moved", "changed" → trigger workflows
- ✅ **Verification primitives** — Re-observe and verify actions worked

**The test question:**
> "What can VisionGraph express naturally that would be awkward in LangGraph + VLM?"

**Answer:**
> "Visual State + observation loops + memory + events + spatial/temporal reasoning as first-class workflow concepts."

---

## Technology Stack

### Core
- **Language:** Python 3.10+
- **Framework:** FastAPI (backend), React (future UI)
- **Testing:** pytest

### Vision
- **VLM:** Phi-3.5-Vision (primary), others via adapters
- **Models:** Hugging Face transformers
- **Core:** PyTorch
- **Tools:** OpenCV, Pillow, NumPy

### Infrastructure
- **Development:** Google Colab (Phase 0-2)
- **Hosting:** Streamlit Cloud (demo), Railway/Fly.io (future)
- **Database:** SQLite (dev), PostgreSQL (future)

---

## Success Metrics (By Phase)

### V0.1
- [ ] Observation → State → Action loop works
- [ ] One demo (desktop login) succeeds
- [ ] Test question answered: "Is abstraction better?"

### V0.2
- [ ] Memory helps reasoning (measurable improvement)
- [ ] Events trigger correctly
- [ ] 50+ GitHub stars

### V0.3
- [ ] Temporal reasoning works
- [ ] Spatial relationships correct
- [ ] 5+ example workflows

### V1.0
- [ ] 500+ GitHub stars
- [ ] 10+ active users
- [ ] Used in real projects
- [ ] Thriving community

---

## Key Decisions Made

### 1. Proof → Expand (Not All-In, Not Tiny)
- ❌ Don't kill the vision (shrink to tiny wrapper)
- ❌ Don't bet everything on V0.1 (try to build full framework immediately)
- ✅ Prove core abstraction, then expand toward vision

### 2. No Entity Tracking in V0.1
- Entity tracking is research-hard and not critical
- V0.1 just needs per-cycle perception
- Can add tracking in V0.2+ if needed

### 3. Framework-First, Not No-Code-First
- V0.1 is a Python SDK (like LangGraph)
- No-code UI comes later (if at all)
- Developers compose workflows in code

### 4. Desktop Automation as First Demo (Not Entire Identity)
- Proves the abstraction works
- Shows practical value
- Foundation for expanding to other domains

### 5. Can Integrate With LangGraph
- Not competing, complementing
- VisionGraph handles visual perception layer
- LangGraph (or other agents) handle higher-level reasoning

---

## Current Status

### ✅ Complete (Phase 0 Prep)
- CORE_THESIS.md — Vision and differentiation
- PLANNING.md — Detailed timeline
- ARCHITECTURE.md — Technical design
- docs/API_SPEC.md — Developer API spec
- docs/EXAMPLES.md — 8 example workflows
- docs/SCHEMAS.md — Data structures
- project.md — Project overview
- README.md — Quick start
- CONTRIBUTING.md — Dev guidelines
- All setup files (setup.py, requirements.txt, .gitignore, LICENSE)

### 🏗️ In Progress (Phase 0 Implementation)
- visiongraph/ — Python package (skeleton started)
- Google Colab notebook (to be created)

### 📋 Not Started (Phase 1+)
- Visual memory system
- Event detection
- Temporal reasoning
- Specialized nodes
- No-code UI

---

## How to Use This Plan

### For Development
1. Read CORE_THESIS.md (understand the vision)
2. Read PLANNING.md (see the timeline)
3. Read docs/API_SPEC.md (see what you're building)
4. Start Phase 0 in Colab (implement V0.1)
5. Ship, learn, move to Phase 1

### For Contributors (Future)
1. Pick a phase you want to work on
2. Reference the phase description
3. Reference docs/API_SPEC.md for expected behavior
4. Submit PR

### For Users (Future)
1. Start with a use case from docs/EXAMPLES.md
2. Reference docs/API_SPEC.md
3. Build your workflow
4. Share results

---

## What's Next

**Immediate (This Week):**
1. ✅ Documentation complete (you're here)
2. → Push to GitHub
3. → Create Google Colab notebook
4. → Start Phase 0 implementation

**Phase 0 (Weeks 1-8):**
- Build observation → perception → state → action → verify loop
- Ship V0.1
- Answer: "Does this abstraction work?"

**Then (Phases 1-4):**
- Expand based on learnings
- Move toward full VisionGraph vision
- Build toward production-ready framework

---

## The Bet You're Making

**Bet:** "Visual State + observation loops + memory + events are useful first-class primitives for visual workflows."

**How you prove it:** Build V0.1, show it's better than alternatives

**If true:** Expand toward full VisionGraph vision (Phases 1-4)  
**If false:** Learn, pivot, ship as reference anyway

---

## The Vision, Restated

**VisionGraph = Vision-native workflow orchestration**

Not just wrapping VLMs. But treating:
- Observation
- Perception
- Visual State
- Memory
- Temporal/Spatial Understanding
- Events
- Verification

As first-class workflow primitives.

That's the framework you're building.

---

**Let's build it.** 🚀

See specific docs for details:
- [CORE_THESIS.md](./CORE_THESIS.md) — Philosophy & differentiation
- [PLANNING.md](./PLANNING.md) — Detailed roadmap
- [ARCHITECTURE.md](./ARCHITECTURE.md) — Technical design
- [docs/API_SPEC.md](./docs/API_SPEC.md) — Complete API
- [docs/EXAMPLES.md](./docs/EXAMPLES.md) — Real workflows
