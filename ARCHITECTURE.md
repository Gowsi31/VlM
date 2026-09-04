# VisionGraph Architecture

## System Overview

**VisionGraph = Orchestration + Observation + Spatial Reasoning + Temporal Understanding + Agent Actions**

```
┌──────────────────────────────────────────────────────────────────┐
│                    VisionGraph Agent                             │
│             (Continuous Visual Observation Loop)                 │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐                                            │
│  │  Visual Input   │                                            │
│  │  • Camera       │                                            │
│  │  • Video        │                                            │
│  │  • Screenshots  │                                            │
│  └────────┬────────┘                                            │
│           │                                                      │
│           ↓                                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Observation Stream                         │   │
│  │  Frame Queue (fps configurable)                         │   │
│  └────────┬────────────────────────────────────────────────┘   │
│           │                                                      │
│  ┌────────┴────────┬────────────────┬──────────────────┐       │
│  ↓                 ↓                ↓                  ↓        │
│ ┌──────────┐  ┌──────────┐   ┌──────────┐   ┌──────────────┐  │
│ │ Vision   │  │ Spatial  │   │Temporal  │   │   Memory     │  │
│ │ Node     │  │ Node     │   │ Node     │   │   Node       │  │
│ │          │  │          │   │          │   │              │  │
│ │ What do  │  │ Where is │   │ How did  │   │ Remember?    │  │
│ │ I see?   │  │ it? How  │   │ it      │   │ Learn?       │  │
│ │          │  │ arranged?│   │ change? │   │              │  │
│ └────┬─────┘  └────┬─────┘   └────┬────┘   └────┬─────────┘  │
│      │             │               │             │             │
│      └─────────────┼───────────────┼─────────────┘             │
│                    │               │                            │
│                    ↓               ↓                            │
│         ┌──────────────────────────────────┐                   │
│         │  Reasoning Node                  │                   │
│         │  LLM Chain of Thought            │                   │
│         └────────────┬─────────────────────┘                   │
│                      │                                          │
│                      ↓                                          │
│         ┌──────────────────────────────────┐                   │
│         │  Decision Node                   │                   │
│         │  What should I do?               │                   │
│         └────┬────────────────────┬────────┘                   │
│              │                    │                             │
│         ┌────▼────┐          ┌────▼────┐                       │
│         │ Action  │          │ Continue │                      │
│         │ Node    │          │ Observe  │                      │
│         └────┬────┘          └────┬─────┘                      │
│              │                    │                             │
│              └────────┬───────────┘                             │
│                       │                                         │
│            ┌──────────▼──────────┐                             │
│            │  Persistent State   │                             │
│            │                     │                             │
│            │ • Visual Obs        │                             │
│            │ • Spatial Map       │                             │
│            │ • Change Log        │                             │
│            │ • Memory (work+long)│                             │
│            │ • Reasoning Chain   │                             │
│            └─────────────────────┘                             │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

             ↑ Loop back (continuous agent cycle) ↓
```

---

## Core Components

### 0. Observation System
**File:** `visiongraph/observation.py`

Manages continuous or discrete visual input.

```python
class ObservationStream:
    def __init__(self, source, fps=2, frame_skip=1):
        self.source = source  # Camera, video, function, etc.
        self.fps = fps
        self.frame_skip = frame_skip
        self.frame_queue = Queue()
        self.history = deque(maxlen=1000)
    
    def start()
    def stop()
    def get_frame() -> np.ndarray
    def get_history(n_frames) -> List[Frame]
```

**Input Sources:**
- Camera: `agent.observe(camera_id=0)`
- Video: `agent.observe("video.mp4")`
- Screenshots: `agent.observe(lambda: pyautogui.screenshot())`
- Function: `agent.observe(get_frame_function)`

---

### 1. Graph Engine
**File:** `visiongraph/graph.py`

Manages the directed acyclic graph (DAG) of nodes.

```python
class VisionGraph:
    def __init__(self, model, config):
        self.model = model
        self.nodes = {}      # {name: Node}
        self.edges = []      # [(source, target, condition)]
        self.state = {}      # Runtime state
    
    def addNode(name, node) -> VisionGraph
    def connect(source, target, condition) -> VisionGraph
    def compile() -> CompiledAgent
    def validate() -> bool
    def visualize() -> str (mermaid diagram)
```

**Key methods:**
- `addNode()` — Register a node
- `connect()` — Create edge with optional condition
- `compile()` — Validate & prepare for execution
- `_topological_sort()` — Ensure valid DAG
- `_validate_graph()` — Check for cycles, orphans

---

### 2. Node System
**File:** `visiongraph/nodes.py`

Base class + specialized node types.

```python
class Node:
    def __init__(self, name, config):
        self.name = name
        self.config = config
    
    def execute(state: GraphState) -> GraphState
    def validate() -> bool
    def get_schema() -> dict
```

**Node Types:**

#### VisionNode
```python
class VisionNode(Node):
    def __init__(self, prompt, output_format="json", vision_tools=[]):
        self.prompt = prompt
        self.output_format = output_format
        self.vision_tools = vision_tools  # ["ocr", "object_detection", ...]
    
    def execute(state):
        # 1. Preprocess image
        # 2. Prepare prompt
        # 3. Call VLM
        # 4. Parse output
        # 5. Update state
```

#### ToolNode
```python
class ToolNode(Node):
    def __init__(self, tool_type, tool_config):
        self.tool_type = tool_type  # "search", "api", "database", "browser"
        self.tool_config = tool_config
    
    def execute(state):
        # 1. Get tool from registry
        # 2. Execute with state data
        # 3. Return result
```

#### DecisionNode
```python
class DecisionNode(Node):
    def __init__(self, logic):
        self.logic = logic  # lambda or string
    
    def execute(state):
        # 1. Evaluate logic
        # 2. Return next node name
```

#### ReasoningNode, ActionNode, HumanNode
Similar pattern...

---

### 3. State Management
**File:** `visiongraph/state.py`

Immutable state object passed through graph.

```python
@dataclass
class GraphState:
    # Input
    image: bytes
    prompt: str
    
    # Visual understanding
    visual_state: VisualState
    
    # LLM reasoning
    reasoning: str
    confidence: float
    
    # Execution
    current_node: str
    step_count: int
    history: List[StepRecord]
    
    # Tools
    tools_used: List[ToolRecord]
    
    # Memory
    memory: Dict[str, Any]
    
    def update(self, **kwargs) -> GraphState:
        # Return new immutable state
        return GraphState(**{**self.__dict__, **kwargs})
    
    def get_visual_state(self) -> VisualState:
        # Objects, scene, errors
        pass
```

**VisualState:**
```python
@dataclass
class VisualState:
    objects: List[DetectedObject]
    scene: SceneDescription
    text_content: TextExtraction
    errors: List[ErrorDetection]
    overall_confidence: float
```

---

### 4. VLM Client
**File:** `visiongraph/vlm_client.py`

Abstraction over different VLM providers.

```python
class VLMClient:
    def __init__(self, model, config):
        self.model = model
        self.provider = self._get_provider()  # Local, OpenAI, etc.
    
    def infer(image, prompt, output_format="text"):
        # 1. Preprocess image
        # 2. Call provider
        # 3. Parse output
        # 4. Cache result
    
    def batch_infer(images, prompts):
        # Efficient batch processing
    
    def stream_infer(image, prompt):
        # Streaming responses for long tasks
```

**Supported Providers:**
- Local (Ollama, LM Studio)
- OpenAI (GPT-4V)
- Anthropic (Claude Vision)
- Hugging Face Inference
- Replicate

---

### 5. Tool Registry
**File:** `visiongraph/tools/`

Pluggable tools for ToolNode.

```
tools/
├── base.py           # BaseTool class
├── search.py         # Google, DuckDuckGo
├── api.py            # HTTP requests
├── database.py       # SQL queries
├── browser.py        # Screenshot, click, type
└── python.py         # Execute Python code
```

Each tool implements:
```python
class BaseTool:
    def execute(input_data) -> output_data
    def validate_input() -> bool
    def get_schema() -> dict
```

---

### 6. Execution Engine
**File:** `visiongraph/executor.py`

Runs the compiled graph.

```python
class Executor:
    def __init__(self, graph, config):
        self.graph = graph
        self.state = GraphState()
        self.max_steps = config.max_steps
    
    def run(image, prompt, **kwargs):
        # 1. Initialize state
        # 2. Start with first node
        # 3. Loop:
        #    a. Execute node
        #    b. Update state
        #    c. Get next node(s)
        #    d. Check termination
        # 4. Return final state
    
    def stream(image, prompt):
        # Yield partial results as they complete
    
    def _execute_node(node, state):
        # Execute single node with error handling
    
    def _get_next_nodes(current_node, state):
        # Evaluate conditions, return next node(s)
```

---

### 7. Memory System
**File:** `visiongraph/memory.py`

Persistent memory across steps.

```python
class Memory:
    def __init__(self):
        self.short_term = {}    # Cleared per execution
        self.long_term = {}     # Persists across executions
        self.context = {}       # Session context
    
    def set(key, value)
    def get(key) -> value
    def add_context(key, value)
    def recall(query) -> List[memory_item]
    def clear()
```

---

### 7.5 Spatial Reasoning System
**File:** `visiongraph/spatial.py`

Builds and maintains a spatial map of the visual scene.

```python
class SpatialNode(Node):
    def __init__(self, task, resolution=1024):
        self.task = task  # "map_objects", "track", "detect_proximity", "path_plan"
        self.resolution = resolution
        self.spatial_grid = None
        self.object_registry = {}
    
    def execute(state: GraphState) -> GraphState:
        # 1. Extract objects from visual state
        # 2. Compute bounding boxes
        # 3. Build spatial grid
        # 4. Calculate relationships (above, below, near, etc.)
        # 5. Update state with spatial map
        
    def map_objects(visual_objects) -> SpatialMap
    def track_movement(current_frame, previous_frame) -> MovementLog
    def detect_proximity(objects) -> ProximityMap
    def plan_path(start, goal) -> Path
```

**SpatialMap Structure:**
```python
class SpatialMap:
    objects_at: Dict[str, BoundingBox]  # {"button_x": [100, 200, 150, 240]}
    spatial_grid: np.ndarray             # 2D grid of object IDs
    relationships: List[Relationship]    # Object-to-object relationships
    quad_tree: QuadTree                  # Fast spatial queries
    
    def get_objects_at(x, y, radius)
    def get_near(obj_id, distance)
    def get_path(from_obj, to_obj)
```

---

### 7.6 Temporal Reasoning System
**File:** `visiongraph/temporal.py`

Analyzes changes over time, detects anomalies, predicts future states.

```python
class TemporalNode(Node):
    def __init__(self, task, memory_frames=30):
        self.task = task  # "detect_changes", "predict", "anomaly_detect", "stability"
        self.memory_frames = memory_frames
        self.frame_history = deque(maxlen=memory_frames)
        self.change_log = []
    
    def execute(state: GraphState) -> GraphState:
        # 1. Compare current frame with previous frames
        # 2. Detect changes (new objects, disappeared, moved, changed)
        # 3. Build temporal context
        # 4. Predict next state
        # 5. Detect anomalies
        
    def detect_changes(frame_t, frame_t1) -> ChangeLog
    def predict_next_frame() -> Prediction
    def detect_anomalies() -> Anomalies
    def detect_stability(threshold=0.95, frames=5) -> bool
```

**ChangeLog Structure:**
```python
class Change:
    timestamp: float
    from_frame: int
    to_frame: int
    type: str  # "appeared", "disappeared", "moved", "changed"
    object_id: str
    magnitude: float  # pixels moved, confidence delta, etc.
    confidence: float

class ChangeLog:
    changes: List[Change]
    significant_events: List[Event]
    stability_score: float  # How much changed (0-1)
```

---

## Data Flow

### Continuous Agent Cycle

```
1. Observation (Read Frame)
   Frame from camera/video
   
2. Vision Node (What do I see?)
   - Detect objects
   - Extract text
   - Classify scene
   - Output: VisualObservation
   
3. Spatial Node (Where is it?)
   - Build spatial map
   - Compute relationships
   - Plan paths
   - Output: SpatialMap
   
4. Temporal Node (What changed?)
   - Compare to previous frames
   - Detect changes
   - Track movements
   - Output: ChangeLog
   
5. Memory Node (Do I remember this?)
   - Query memories
   - Store observations
   - Output: MemoryRecall
   
6. Reasoning Node (What does this mean?)
   - Chain of thought
   - LLM analysis
   - Output: Reasoning
   
7. Decision Node (What should I do?)
   - Evaluate policy
   - Output: Decision ("act", "observe", "help", etc.)
   
8. Action Node (Do it)
   - Click, type, search, wait
   - Output: ActionResult
   
9. State Update
   - Update persistent state
   - Loop back to observation
```

### Single Step Execution

```
1. Input
   GraphState(image, prompt)
   
2. Node Execution
   VisionNode.execute(state)
   
3. VLM Call
   VLMClient.infer(image, prompt)
   
4. Output Processing
   Parse response → VisualState
   
5. State Update
   state.visual_state = new_objects
   state.confidence = 0.92
   state.reasoning = "Found error on line 42"
   
6. Decision
   DecisionNode.execute(state)
   → Returns: "next_node_name"
   
7. Next Step
   Fetch next_node from graph
   Execute with updated state
```

---

## Configuration Hierarchy

```
Default Config (code)
    ↓
graph = VisionGraph(model="phi-3.5-vision")
    ↓
User Config (env vars, config files)
    ↓
Runtime Config (kwargs)
    ↓
Final Config (used for execution)
```

**Example:**
```python
# Defaults
config.temperature = 0.7
config.max_tokens = 1024

# User overrides
config.temperature = 0.9  # More creative

# Runtime
agent.run(image, temperature=0.5)  # One-off override
```

---

## Error Handling

```
VisionNodeError
├── ImageProcessingError
├── InferenceError
└── OutputParsingError

ToolNodeError
├── ToolNotFoundError
├── ToolExecutionError
└── ToolTimeoutError

GraphError
├── InvalidGraphError
├── MaxStepsExceededError
├── StateCorruptionError
└── NodeExecutionError
```

Each error includes:
- Error type
- Human-readable message
- Context (which node, step number)
- Recovery suggestions

---

## Performance Considerations

### Image Optimization
- Resize large images (downscale to 1024x1024)
- JPEG compression for faster inference
- Cache processed images

### Inference Batching
- Group multiple inferences
- Parallel execution where possible

### State Caching
- LRU cache for VLM responses
- Memcached for distributed setup

### Monitoring
- Step execution time
- VLM latency
- Memory usage
- Tool execution time

---

## Extension Points

### Custom Nodes
```python
class CustomNode(Node):
    def execute(self, state: GraphState) -> GraphState:
        # Custom logic
        return state.update(...)
```

### Custom Tools
```python
class CustomTool(BaseTool):
    def execute(self, input_data):
        # Tool logic
        return output_data
```

### Custom VLM Providers
```python
class CustomProvider(VLMProvider):
    def infer(self, image, prompt):
        # Provider logic
        return response
```

---

## Testing Strategy

```
Unit Tests (nodes, state, tools)
    ↓
Integration Tests (graph execution)
    ↓
End-to-End Tests (real workflows)
    ↓
Performance Tests (latency, memory)
```

---

## Deployment

### Single Machine
- Python process + API server
- SQLite for state
- Local VLM (Ollama)

### Cloud
- FastAPI service (scalable)
- PostgreSQL for state
- Managed VLM inference (Replicate, HF)
- Redis for caching

---

See [API_SPEC.md](./docs/API_SPEC.md) for complete API reference.
