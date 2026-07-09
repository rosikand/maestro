from abc import ABC, abstractmethod


class Model(ABC):

    @abstractmethod
    def generate(
        self,
        prompts,
        n=1,
        temperature=1.0,
        max_tokens=256,
    ):
        pass
