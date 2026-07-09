from maestro.tasks import get_task


def test_countdown_repr():
    task = get_task("countdown", split="train", limit=5)

    assert "CountdownTask" in repr(task)
    assert "split='train'" in repr(task)
    assert "limit=5" in repr(task)
    assert "examples=5" in repr(task)

    assert "CountdownTask:" in str(task)
    assert "arithmetic expression" in str(task)
    assert "5 examples" in str(task)


def test_countdown_reward():
    task = get_task("countdown", limit=1)

    example = {
        "prompt": "test",
        "nums": [19, 74, 45],
        "target": 48,
    }

    assert task.reward(
        "<answer>74 + 19 - 45</answer>",
        example,
    ) == 1.0

    assert task.reward(
        "<answer>74 - 45</answer>",
        example,
    ) == 0.0

    assert task.reward(
        "<answer>48</answer>",
        example,
    ) == 0.0
