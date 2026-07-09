"""Countdown numbers game (as in TinyZero) for RL post-training.

Puzzles are generated locally: the target is built by combining the numbers,
so every puzzle is solvable and no download is needed.
"""

import random
import re

from datasets import Dataset

PROMPT_TEMPLATE = (
    "Using the numbers {numbers}, create an equation that equals {target}. "
    "You may use +, -, *, / and each number exactly once. "
    "Write the final equation in <answer> </answer> tags, e.g. <answer>(1 + 2) / 3</answer>."
)

_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_VALID_EXPR = re.compile(r"^[\d+\-*/() ]+$")


def load(n: int = 1000, num_count: int = 4, seed: int = 0) -> Dataset:
    """Generate `n` solvable puzzles: `prompt`, `numbers`, `target`, and one known `solution`."""
    rng = random.Random(seed)
    return Dataset.from_list([_make_example(rng, num_count) for _ in range(n)])


def _make_example(rng: random.Random, num_count: int) -> dict:
    while True:
        numbers = [rng.randint(1, 99) for _ in range(num_count)]
        target, solution = numbers[0], str(numbers[0])
        for n in numbers[1:]:
            ops = ["+", "*"]
            if target > n:  # keep intermediate values positive integers
                ops.append("-")
            if target % n == 0:
                ops.append("/")
            op = rng.choice(ops)
            if op == "+":
                target += n
            elif op == "-":
                target -= n
            elif op == "*":
                target *= n
            else:
                target //= n
            solution = f"({solution} {op} {n})"
        if 1 <= target <= 999:
            shuffled = rng.sample(numbers, len(numbers))  # hide the solution order
            return {
                "prompt": PROMPT_TEMPLATE.format(numbers=shuffled, target=target),
                "numbers": shuffled,
                "target": target,
                "solution": solution,
            }


def extract_expression(text: str) -> str | None:
    """The equation in the last <answer> block, keeping only the left side of any `=`."""
    matches = _ANSWER_TAG.findall(text)
    return matches[-1].split("=")[0].strip() if matches else None


def reward(completions, numbers, target, **kwargs) -> list[float]:
    """1.0 per completion whose equation uses each number exactly once and hits the target."""
    scores = []
    for completion, nums, tgt in zip(completions, numbers, target):
        if isinstance(completion, list):  # chat format: [{"role": ..., "content": ...}]
            completion = completion[-1]["content"]
        scores.append(_score(extract_expression(completion), nums, tgt))
    return scores


def _score(expr: str | None, numbers: list[int], target: int) -> float:
    if expr is None or not _VALID_EXPR.match(expr) or "**" in expr or "//" in expr:
        return 0.0
    if sorted(int(x) for x in re.findall(r"\d+", expr)) != sorted(numbers):
        return 0.0
    try:
        value = eval(expr)  # safe: charset above allows only digits, + - * / ( ) and spaces
    except Exception:
        return 0.0
    return 1.0 if abs(value - target) < 1e-6 else 0.0
