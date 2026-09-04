# VisionGraph Project Overview

## What is VisionGraph?

VisionGraph is a **framework for building visual agents with continuous observation, spatial reasoning, and temporal understanding**.

Think of it like **LangGraph for robotics/desktop automation** — but for any visual domain.

**VisionGraph = Observation + Spatial Reasoning + Temporal Understanding + Agent Actions**

Instead of single-frame analysis:
```python
llm_response = model.chat("What do you see in this image?")
```

You build continuous agents:
```python
agent = VisionGraph(model="phi-3.5-vision")
agent.observe(camera_feed)  # Continuous visual input

agent.add_node("vision", VisionNode(prompt="What do you see?"))
agent.add_node("spatial", SpatialNode(task="map_objects"))
agent.add_node("temporal", TemporalNode(task="detect_changes"))
agent.add_node("reason", ReasoningNode(prompt="What happened? What to do?"))
agent.add_node("decide", DecisionNode())
agent.add_node("act", ActionNode(action_type="click"))

agent.connect("observe", "vision").connect("vision", "spatial")\
    .connect("spatial", "temporal").connect("temporal", "reason")\
    .connect("reason", "decide").connect("decide", "act")\
    .connect("act", "observe")  # Loop back

agent.run()  # Continuous operation
```

---

## Why VisionGraph?

1. **Orchestration** — Manage complex VLM workflows
2. **Reusability** — Define agents once, run anywhere
3. **State Management** — Track what the model sees (visual state)
4. **Tool Integration** — Connect to APIs, databases, tools
5. **Extensibility** — Custom nodes, plugins, VLM swaps
6. **Open Source** — No vendor lock-in

---

## Core Concepts

### Graph
A directed acyclic graph (DAG) of connected nodes representing an agent workflow.

### Nodes
Processing units that transform data:
- **VisionNode** — Analyze images with VLM
- **ToolNode** — Call external services
- **ReasoningNode** — LLM reasoning
- **DecisionNode** — Branch logic
- **ActionNode** — Execute final action
- **HumanNode** — Pause for approval

### State
Shared context flowing through the graph:
- Current image/visual input
- Visual state (detected objects, coordinates)
- Reasoning/text from LLM
- Tool outputs
- Memory

### Edges
Connections between nodes with optional conditions.

---

## Project Structure

```
visiongraph/
├── visiongraph/           # Main package
│   ├── __init__.py
│   ├── graph.py           # Graph class
│   ├── nodes.py           # Node types
│   ├── state.py           # State management
│   ├── vlm_client.py      # VLM inference
│   ├── tools/
│   │   ├── search.py
│   │   ├── browser.py
│   │   ├── database.py
│   │   └── api.py
│   └── utils/
│       ├── config.py
│       ├── logging.py
│       └── cache.py
├── docs/                  # Documentation
│   ├── README.md
│   ├── ARCHITECTURE.md
│   ├── API_SPEC.md
│   ├── EXAMPLES.md
│   └── SCHEMAS.md
├── examples/              # Example workflows
│   ├── screenshot_analyzer.py
│   ├── ui_tester.py
│   ├── desktop_agent.py
│   └── doc_generator.py
├── tests/                 # Unit tests
│   ├── test_graph.py
│   ├── test_nodes.py
│   └── test_vlm_client.py
├── notebooks/             # Colab notebooks
│   ├── quick_start.ipynb
│   ├── development.ipynb
│   └── examples.ipynb
├── README.md
├── LICENSE
├── setup.py
├── requirements.txt
├── .gitignore
├── project.md             # This file
└── CONTRIBUTING.md
```

---

## Development Phases

### Phase 1: Design ✅ (Complete)
- [x] API specification
- [x] Example workflows
- [x] Data schemas
- [x] Architecture diagrams

### Phase 2: Core Framework (Week 2)
- [ ] Graph class (orchestration engine)
- [ ] Node types (VisionNode, ToolNode, etc.)
- [ ] State management
- [ ] Mock VLM client
- [ ] Basic examples
- [ ] Unit tests

### Phase 3: Web UI (Week 3)
- [ ] Streamlit app
- [ ] Graph visualization
- [ ] Workflow playground
- [ ] Deploy to cloud

### Phase 4: Real VLM (Week 4)
- [ ] Phi-3.5-Vision integration
- [ ] Performance benchmarking
- [ ] Integration tests

### Phase 5: Release (Week 5+)
- [ ] PyPI packaging
- [ ] Documentation site
- [ ] Example gallery
- [ ] v0.1 public release

---

## Quick Start

### Installation (Future)
```bash
pip install visiongraph
```

### Basic Usage
```python
from visiongraph import VisionGraph, VisionNode, ActionNode

# Create graph
graph = VisionGraph(model="phi-3.5-vision")

# Add nodes
graph.addNode("analyze", VisionNode(
    prompt="What's in this image?"
))
graph.addNode("output", ActionNode())

# Connect
graph.connect("analyze", "output")

# Run
agent = graph.compile()
result = agent.run(image="screenshot.png")
print(result)
```

---

## Current Status

**Phase 1:** Design documentation complete  
**Phase 2:** Starting this week in Colab

---

## Tech Stack

**Core:**
- Python 3.10+
- FastAPI (backend)
- React (frontend)

**VLM:**
- Hugging Face transformers
- Ollama (local inference)
- Phi-3.5-Vision (initial model)

**Infrastructure:**
- Google Colab (GPU testing)
- Streamlit (web UI)
- SQLite (state storage)

**Tools:**
- pytest (testing)
- Sphinx (docs)

---

## Contributors

- You (Gowsalya)
- Claude (AI assistant)

---

## License

MIT (open source)

---

## Next Steps

1. ✅ Create GitHub repo
2. ✅ Design & documentation  
3. → **Start Phase 2: Core framework in Colab**
4. Build web UI
5. Integrate real VLM
6. Release v0.1

See [ARCHITECTURE.md](./docs/ARCHITECTURE.md) for technical details.
