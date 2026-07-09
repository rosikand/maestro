# Models

A model wraps an inference backend behind one method: `generate`. Get one from the registry:

```python
from maestro.models import get_model

model = get_model("Qwen/Qwen3-1.7B")
```

- `model_name` is a Hugging Face model id.
- `backend` picks the inference backend (default `"transformers"`, currently the only one).
- Extra keyword arguments are forwarded to the backend (e.g. `device_map`, `dtype` for
  the transformers backend).

## Generating

`generate` takes a single prompt or a list of prompts. Prompts are plain user strings —
the backend applies the model's chat template for you:

```python
output = model.generate("What is 2 + 2?", max_tokens=64)

outputs = model.generate(
    ["What is 2 + 2?", "Name a prime number."],
    temperature=1.0,
    max_tokens=256,
)
```

- `n` returns multiple samples per prompt (requires `temperature > 0`).
- `temperature=0` selects greedy decoding.
- A single string prompt with `n=1` returns a string; otherwise a list of completions.

## With a task

Models and tasks compose into the end-to-end loop:

```python
from maestro.models import get_model
from maestro.tasks import get_task

model = get_model("Qwen/Qwen3-1.7B")
task = get_task("countdown", limit=10)

example = task[0]
completion = model.generate(example["prompt"], temperature=1.0, max_tokens=256)
reward = task.reward(completion, example)
```

## Available backends

| Backend | Class | Notes |
|---|---|---|
| `transformers` | `TransformersModel` | Hugging Face causal LM with `device_map="auto"` and `dtype="auto"`; left-padded batching; chat template applied to prompts |

## Adding a backend

Subclass `Model`, implement `generate()`, and register it:

```python
# src/maestro/models/registry.py
BACKENDS = {
    "transformers": TransformersModel,
    "mybackend": MyBackendModel,
}
```
