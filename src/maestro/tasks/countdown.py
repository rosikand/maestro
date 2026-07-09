import ast
import operator
import re
from collections import Counter
from fractions import Fraction

from datasets import load_dataset

from .base import Task


OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def evaluate_expression(expression):
    tree = ast.parse(expression, mode="eval")
    used_numbers = []

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)

        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, int)
            and not isinstance(node.value, bool)
        ):
            used_numbers.append(node.value)
            return Fraction(node.value)

        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            left = evaluate(node.left)
            right = evaluate(node.right)

            return OPS[type(node.op)](left, right)

        raise ValueError("Invalid expression")

    value = evaluate(tree)

    return value, used_numbers


def extract_expression(completion):
    match = re.search(
        r"<answer>(.*?)</answer>",
        completion,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if match:
        return match.group(1).strip()

    lines = [
        line.strip()
        for line in completion.splitlines()
        if line.strip()
    ]

    if not lines:
        return ""

    return lines[-1]


class CountdownTask(Task):

    description = (
        "Small dataset of arithmetic expressions to solve. Given a set of numbers, produce an arithmetic expression "
        "that equals a target using each number exactly once. Use each number exactly once. You may use +, -, *, /, "
        "and parentheses. Put the final expression inside <answer>...</answer>."
    )

    def load_dataset(self):
        dataset = load_dataset(
            "Jiayi-Pan/Countdown-Tasks-3to4",
            split=self.split,
        )

        def format_example(example):
            return {
                "prompt": (
                    f"Using the numbers {example['nums']}, create an "
                    f"arithmetic expression equal to {example['target']}. "
                    f"Use each number exactly once. You may use +, -, *, /, "
                    f"and parentheses. Put the final expression inside "
                    f"<answer>...</answer>."
                )
            }

        return dataset.map(format_example)

    def reward(self, completion, example):
        expression = extract_expression(completion)

        if not expression:
            return 0.0

        try:
            value, used_numbers = evaluate_expression(expression)

            numbers_match = (
                Counter(used_numbers)
                == Counter(example["nums"])
            )

            target_match = (
                value
                == Fraction(example["target"])
            )

            return float(numbers_match and target_match)

        except (
            SyntaxError,
            ValueError,
            ZeroDivisionError,
        ):
            return 0.0