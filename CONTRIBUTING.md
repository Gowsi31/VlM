# Contributing to VisionGraph

Thanks for your interest in contributing! This guide will help you get started.

## Getting Started

### 1. Fork & Clone
```bash
git clone https://github.com/YOU/visiongraph.git
cd visiongraph
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install in Development Mode
```bash
pip install -e ".[dev]"
```

### 4. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

## Development Workflow

### Running Tests
```bash
# All tests
pytest tests/

# Specific test
pytest tests/test_graph.py -v

# With coverage
pytest --cov=visiongraph tests/
```

### Code Style
```bash
# Format code
black visiongraph/ tests/

# Lint
flake8 visiongraph/ tests/

# Type checking
mypy visiongraph/
```

### Building Documentation
```bash
cd docs/
sphinx-build -b html . _build/
```

## Code Guidelines

### Style
- Follow PEP 8
- Use type hints
- Write docstrings for public methods
- Max line length: 100 characters

### Example:
```python
def execute(self, state: GraphState) -> GraphState:
    """Execute node with given state.
    
    Args:
        state: Current graph state
    
    Returns:
        Updated graph state
    """
    # Implementation
    return updated_state
```

### Testing
- Write tests for new features
- Aim for >80% coverage
- Use descriptive test names
- Test both success and failure cases

### Example:
```python
def test_vision_node_executes_successfully():
    node = VisionNode(prompt="test")
    state = GraphState(image=test_image)
    result = node.execute(state)
    
    assert result.confidence > 0.0
    assert result.visual_state is not None

def test_vision_node_handles_invalid_image():
    node = VisionNode(prompt="test")
    state = GraphState(image=None)
    
    with pytest.raises(VisionNodeError):
        node.execute(state)
```

## Commit Guidelines

### Format
```
<type>: <subject>

<body>

<footer>
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style (formatting, missing semicolons, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `test`: Adding or updating tests
- `chore`: Build, dependencies, tools

### Example
```
feat: add streaming support to graph execution

Add support for streaming partial results as nodes complete.
Implement stream() method on Executor class.

Fixes #123
```

## Pull Request Process

1. **Create PR** with clear title and description
2. **Link issues** with "Fixes #123"
3. **Run tests** locally before pushing
4. **Add tests** for new functionality
5. **Update docs** if needed
6. **Keep commits clean** - squash if necessary
7. **Wait for review** - address feedback

### PR Template
```markdown
## Description
What does this PR do?

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation

## Testing
How was this tested?

## Checklist
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] No breaking changes
```

## Areas to Contribute

### High Priority
- [ ] Core framework implementation (Phase 2)
- [ ] Node types (Vision, Tool, Decision)
- [ ] Unit tests
- [ ] Bug fixes

### Medium Priority
- [ ] Web UI (Streamlit)
- [ ] Documentation
- [ ] Examples
- [ ] Performance optimization

### Lower Priority
- [ ] Additional VLM support
- [ ] Advanced features
- [ ] Community tools

## Questions?

- Open an issue for bugs
- Use Discussions for questions
- Check existing issues first

## Code of Conduct

- Be respectful
- Constructive feedback only
- Focus on ideas, not people
- Inclusive environment

---

Thank you for contributing! 🎉
