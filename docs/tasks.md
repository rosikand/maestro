# Tasks

A task bundles the two things a post-training loop needs: a dataset with a ready-to-train
`prompt` column, and a `reward` function for scoring completions. Get one from the registry:

```python
from maestro.tasks import get_task

task = get_task("countdown", split="train", limit=1000)
```

- `split` picks the dataset split (default `"train"`).
- `limit` caps the number of examples (useful for quick experiments).

## Using a task

A task acts like a dataset — index, iterate, or take `len()` of it. Each example has a
`prompt` column plus the raw fields of the underlying dataset:

```python
len(task)          # 1000
task[0]["prompt"]  # "Using the numbers [44, 19, 35], create an arithmetic expression..."
task[0]["nums"], task[0]["target"]
```

`task.reward(completion, example)` returns a float score for a single completion:

```python
task.reward("<answer>(44 - 19) + 35</answer>", task[0])  # 1.0 if correct, else 0.0
```

For batched scoring (e.g. inside a trainer), zip completions with their examples:

```python
rewards = [
    task.reward(completion, example)
    for completion, example in zip(completions, examples)
]
```

## Available tasks

| Name | Dataset | Reward |
|---|---|---|
| `countdown` | [Jiayi-Pan/Countdown-Tasks-3to4](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4) | 1.0 if the expression in `<answer>...</answer>` uses each number exactly once and equals the target, else 0.0 |

## Adding a task

Subclass `Task` and implement `load_dataset()` (return a HF dataset with a `prompt` column)
and `reward(completion, example)`, then register it:

```python
# src/maestro/tasks/mytask.py
from datasets import load_dataset
from .base import Task

class MyTask(Task):
    def load_dataset(self):
        dataset = load_dataset("org/my-dataset", split=self.split)
        return dataset.map(lambda x: {"prompt": f"Solve: {x['question']}"})

    def reward(self, completion, example):
        return float(completion.strip() == example["answer"])
```

```python
# src/maestro/tasks/registry.py
TASKS = {
    "countdown": CountdownTask,
    "mytask": MyTask,
}
```
