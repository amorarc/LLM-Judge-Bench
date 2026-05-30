from abc import ABC, abstractmethod
from typing import List


class BaseModel(ABC):
    def __init__(self, model_id: str):
        self.model_id = model_id

    @abstractmethod
    def logprobs(self, text: str) -> float:
        """Mean log-probability per token (non-positive scalar)."""
        ...

    def batch_logprobs(self, texts: List[str]) -> List[float]:
        return [self.logprobs(t) for t in texts]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.model_id!r})"
