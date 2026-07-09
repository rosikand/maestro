from .countdown import CountdownTask


TASKS = {
    "countdown": CountdownTask,
}


def get_task(name, **kwargs):
    if name not in TASKS:
        raise ValueError(f"Unknown task: {name}")

    return TASKS[name](**kwargs)