from maestro.datasets import countdown, gsm8k


def test_gsm8k_extract_answer():
    assert gsm8k.extract_answer("6 + 6 = 12\n#### 12") == "12"
    assert gsm8k.extract_answer("#### 1,200") == "1200"
    assert gsm8k.extract_answer("the answer is 42.") == "42"
    assert gsm8k.extract_answer("no numbers here") is None


def test_gsm8k_reward():
    completions = [
        "6 * 12 = 72\n#### 72",
        "#### 71",
        [{"role": "assistant", "content": "so it must be 72"}],
        "i refuse to answer",
    ]
    assert gsm8k.reward(completions=completions, answer=["72"] * 4) == [1.0, 0.0, 1.0, 0.0]


def test_countdown_load_is_deterministic_and_solvable():
    ds = countdown.load(n=20, num_count=3, seed=0)
    assert ds.to_list() == countdown.load(n=20, num_count=3, seed=0).to_list()
    for row in ds:
        assert 1 <= row["target"] <= 999
        assert len(row["numbers"]) == 3
        solved = f"<answer>{row['solution']}</answer>"
        assert countdown.reward(
            completions=[solved], numbers=[row["numbers"]], target=[row["target"]]
        ) == [1.0]


def test_countdown_reward():
    numbers, target = [2, 3, 7], 42
    good = ["<answer>2 * 3 * 7</answer>", "<answer>(3 * 2) * 7 = 42</answer>"]
    assert countdown.reward(
        completions=good, numbers=[numbers] * 2, target=[target] * 2
    ) == [1.0, 1.0]

    bad = [
        "<answer>6 * 7</answer>",  # 6 is not one of the numbers
        "<answer>2 * 3 * 7 * 1</answer>",  # invents an extra number
        "<answer>2 + 3 + 7</answer>",  # wrong value
        "<answer>2 * 3 * 7",  # unclosed tag
        "2 * 3 * 7",  # no tags
        "<answer>2 ** 3 ** 7</answer>",  # power operator is banned
        "<answer>__import__('os').getcwd()</answer>",  # not arithmetic
    ]
    assert countdown.reward(
        completions=bad, numbers=[numbers] * len(bad), target=[target] * len(bad)
    ) == [0.0] * len(bad)
