# maestro

A minimal post-training library.

## Install

```bash
pip install git+https://github.com/rosikand/maestro.git
```

Pin a tag or commit for reproducibility:

```bash
pip install git+https://github.com/rosikand/maestro.git@v0.0.1
```

Confirm it worked:

```bash
python -c "import maestro; maestro.hello()"
# hello, this is maestro
```

## Datasets

Each dataset module has two functions — everything GRPO needs:

- `load()` → a HF `datasets.Dataset` with a ready-to-train `prompt` column, plus the raw
  fields (so you can re-template or inspect).
- `reward(completions, **columns)` → `list[float]`. This follows the TRL reward-function
  convention, so it works with `GRPOTrainer` as-is, or in a hand-rolled loop.

```python
from maestro.datasets import countdown, gsm8k

train = gsm8k.load("train")               # columns: prompt, question, answer
puzzles = countdown.load(n=5000, seed=0)  # columns: prompt, numbers, target, solution

gsm8k.reward(completions=["6 * 12 = 72\n#### 72"], answer=["72"])  # [1.0]
countdown.reward(completions=["<answer>(1 + 2) / 3</answer>"], numbers=[[1, 2, 3]], target=[1])  # [1.0]
```

With TRL:

```python
from trl import GRPOTrainer

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-1.5B",
    train_dataset=countdown.load(n=5000),
    reward_funcs=countdown.reward,
)
trainer.train()
```

Prompts are plain strings, which is what you want for base models. For instruct/chat models,
wrap them into messages and TRL applies the chat template:

```python
ds = ds.map(lambda x: {"prompt": [{"role": "user", "content": x["prompt"]}]})
```

Countdown puzzles are generated locally with a seed (no download), and the target is built by
combining the numbers, so every puzzle is guaranteed solvable. Use a different seed for a held-out
eval split. GSM8K's eval split is `gsm8k.load("test")`.

## Develop

```bash
pip install -e ".[dev]"
pytest
```
