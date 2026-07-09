"""GSM8K (grade-school math) for RL post-training."""

import re

from datasets import Dataset, load_dataset

PROMPT_TEMPLATE = (
    "{question}\n\n"
    "Think step by step, then give your final answer on the last line as: #### <answer>"
)

_MARKED_ANSWER = re.compile(r"####\s*\$?(-?\d[\d,]*(?:\.\d+)?)")
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def load(split: str = "train") -> Dataset:
    """Load GSM8K with a ready-to-train `prompt` and the gold `answer` as a bare number."""
    ds = load_dataset("openai/gsm8k", "main", split=split)
    return Dataset.from_list(
        [
            {
                "prompt": PROMPT_TEMPLATE.format(question=question),
                "question": question,
                "answer": extract_answer(solution),
            }
            for question, solution in zip(ds["question"], ds["answer"])
        ]
    )


def extract_answer(text: str) -> str | None:
    """The number after the last `#### ` marker, else the last number anywhere."""
    matches = _MARKED_ANSWER.findall(text) or _NUMBER.findall(text)
    return matches[-1].replace(",", "") if matches else None


def reward(completions, answer, **kwargs) -> list[float]:
    """1.0 per completion whose final answer equals the gold answer, else 0.0."""
    scores = []
    for completion, gold in zip(completions, answer):
        if isinstance(completion, list):  # chat format: [{"role": ..., "content": ...}]
            completion = completion[-1]["content"]
        predicted = extract_answer(completion)
        try:
            scores.append(1.0 if float(predicted) == float(gold) else 0.0)
        except (TypeError, ValueError):
            scores.append(0.0)
    return scores
