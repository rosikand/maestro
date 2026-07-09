from abc import ABC, abstractmethod


class Task(ABC):

    description = ""

    def __init__(self, split="train", limit=None):
        self.split = split
        self.limit = limit
        self.dataset = self.load_dataset()

        if limit is not None:
            self.dataset = self.dataset.select(
                range(min(limit, len(self.dataset)))
            )

    @abstractmethod
    def load_dataset(self):
        pass

    @abstractmethod
    def reward(self, completion, example):
        """
        Calculate the reward for a given completion and example.

        Args:
            completion (str): what the model generated
            example (dict): what problem it was answering

        Returns:
            float: The reward for the completion.

        Note:
            For batching with a trainer, use the following pattern: 
            rewards = [
                task.reward(completion, example)
                for completion, example in zip(completions, examples)
            ]
        """
        pass

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __repr__(self):
        limit_part = f", limit={self.limit}" if self.limit is not None else ""
        return (
            f"{type(self).__name__}(split={self.split!r}{limit_part}, "
            f"examples={len(self)})"
        )

    def __str__(self):
        name = type(self).__name__
        if self.description:
            return (
                f"{name}: {self.description} "
                f"({len(self)} examples, split={self.split!r})"
            )
        return repr(self)

    def _apply_limit(self, dataset):
        if self.limit is None:
            return dataset

        limit = min(self.limit, len(dataset))
        return dataset.select(range(limit))