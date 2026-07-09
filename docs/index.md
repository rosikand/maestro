# maestro

A minimal post-training library.

maestro gives you **tasks**: small bundles of everything a post-training loop needs —
a dataset with a ready-to-train `prompt` column, and a `reward` function for scoring
model completions.

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

## Quickstart

```python
from maestro.tasks import get_task

task = get_task("countdown", split="train", limit=1000)

task[0]["prompt"]
# "Using the numbers [44, 19, 35], create an arithmetic expression equal to 98. ..."

task.reward("<answer>(44 + 19) + 35</answer>", task[0])
# 1.0 if correct, else 0.0
```

See [Tasks](tasks.md) for the full guide, and the [API Reference](api.md) for details.

## Develop

```bash
git clone https://github.com/rosikand/maestro.git
cd maestro
pip install -e ".[dev]"
pytest
```

To work on these docs:

```bash
pip install -e ".[docs]"
mkdocs serve
```
