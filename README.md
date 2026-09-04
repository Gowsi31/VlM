# VisionGraph

A graph-based framework for building visual agents with Vision-Language Models.

**Think LangGraph, but for vision.**

```python
from visiongraph import VisionGraph, VisionNode, SpatialNode, TemporalNode, DecisionNode, ActionNode

# Create continuous visual agent
agent = VisionGraph(model="phi-3.5-vision")

# Set visual input (camera/video/screenshots)
agent.observe(lambda: pyautogui.screenshot())

# Perception pipeline
agent.add_node("vision", VisionNode(prompt="What do you see?"))
agent.add_node("spatial", SpatialNode(task="map_objects"))
agent.add_node("temporal", TemporalNode(task="detect_changes"))
agent.add_node("decide", DecisionNode(logic=lambda s: "act" if s.confidence > 0.8 else "observe"))
agent.add_node("act", ActionNode(action_type="click"))

# Orchestration
agent.connect("observe", "vision")
agent.connect("vision", "spatial")
agent.connect("spatial", "temporal")
agent.connect("temporal", "decide")
agent.connect("decide", "act")
agent.connect("act", "observe")  # Loop back for continuous operation

# Run continuously
agent.compile(mode="continuous")
agent.run()
```

## Features

✅ **Continuous observation** — Real-time visual streams (camera, video, screenshots)  
✅ **Spatial reasoning** — Understand object positions, layouts, relationships  
✅ **Temporal understanding** — Detect changes, predict future states, track movements  
✅ **Graph orchestration** — Define complex visual agent workflows  
✅ **Visual state tracking** — Persistent understanding of the scene  
✅ **Tool integration** — Connect to APIs, databases, search, browser  
✅ **Multiple VLMs** — Local (Phi, LLaVA) or cloud (GPT-4V, Claude)  
✅ **Memory system** — Working & long-term memory across episodes  
✅ **Error handling** — Graceful failures & recovery  

## Quick Start

### Installation
```bash
pip install visiongraph
```

### Basic Example
```python
from visiongraph import VisionGraph, VisionNode, ActionNode

graph = VisionGraph(model="phi-3.5-vision")

# Add node
graph.addNode("analyze", VisionNode(
    prompt="Describe what you see"
))
graph.addNode("output", ActionNode())

# Connect
graph.connect("analyze", "output")

# Execute
agent = graph.compile()
result = agent.run(image="image.png")
print(result)
```

## Documentation

- **[Architecture](./ARCHITECTURE.md)** — Technical deep dive
- **[API Reference](./docs/API_SPEC.md)** — Complete API
- **[Examples](./docs/EXAMPLES.md)** — 5 real workflows
- **[Schemas](./docs/SCHEMAS.md)** — Data structures
- **[Project](./project.md)** — Project status & roadmap

## Examples

### Screenshot Error Analysis
Automatically analyze screenshots and investigate errors.

```python
graph = VisionGraph(model="phi-3.5-vision")

graph.addNode("analyze", VisionNode(
    prompt="Is there an error message? If yes, what is it?"
))
graph.addNode("search", ToolNode(tool_type="search"))
graph.addNode("solve", VisionNode(
    prompt="Based on the search results, what's the solution?"
))

graph.connect("analyze", "search")
graph.connect("search", "solve")

agent = graph.compile()
result = agent.run(image=screenshot)
```

### UI Testing
Visual regression testing.

```python
graph = VisionGraph(model="phi-3.5-vision")

graph.addNode("capture", VisionNode(prompt="Analyze this UI"))
graph.addNode("compare", ToolNode(tool_type="database"))
graph.addNode("report", ActionNode())

graph.connect("capture", "compare")
graph.connect("compare", "report")

agent = graph.compile()
result = agent.run(image=current_screenshot)
```

See [Examples](./docs/EXAMPLES.md) for more workflows.

## Supported Models

### Local
- **Phi-3.5-Vision** (4B, recommended)
- **LLaVA-1.6-7B** (7B)
- **Qwen2-VL-7B** (7B)

### Cloud
- **GPT-4V** (OpenAI)
- **Claude Vision** (Anthropic)
- **Gemini Pro Vision** (Google)

Configure via:
```python
graph = VisionGraph(model="phi-3.5-vision")  # Local
graph = VisionGraph(model="gpt-4-vision", api_key="...")  # OpenAI
```

## Installation Options

### Option 1: Minimal (CPU only)
```bash
pip install visiongraph
```

### Option 2: GPU Support (NVIDIA)
```bash
pip install visiongraph[cuda]
```

### Option 3: Development
```bash
git clone https://github.com/YOU/visiongraph.git
cd visiongraph
pip install -e ".[dev]"
```

## Development

### Running Tests
```bash
pytest tests/
pytest tests/test_graph.py -v
```

### Building Locally
```bash
pip install -e .
```

### Running Examples
```bash
python examples/screenshot_analyzer.py
python examples/ui_tester.py
```

## Architecture

```
Image → VisionNode → VisualState
           ↓
      DecisionNode
           ↓
      ToolNode (Search/API)
           ↓
      ReasoningNode
           ↓
      ActionNode → Result
```

See [ARCHITECTURE.md](./ARCHITECTURE.md) for full details.

## Status

🚀 **v0.1 - Early Development**

- Phase 1: ✅ Design & documentation
- Phase 2: 🏗️ Core framework (Colab)
- Phase 3: 🏗️ Web UI (Streamlit)
- Phase 4: 🏗️ Real VLM integration
- Phase 5: 🏗️ v0.1 Release

Roadmap: [project.md](./project.md)

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](./LICENSE) file

## Support

- 📖 Read the [docs](./docs)
- 💬 GitHub Issues
- 📧 Email: [your email]

## Acknowledgments

Inspired by:
- [LangGraph](https://github.com/langchain-ai/langgraph) — Agent orchestration
- [ComfyUI](https://github.com/comfyorg/ComfyUI) — Node-based workflows
- [PyVision](https://github.com/agents-x-project/PyVision) — Dynamic tooling

---

**Built with:** Python, Hugging Face, Ollama, FastAPI, React
