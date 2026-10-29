from dataclasses import dataclass, field
from typing import Self

import numpy as np


@dataclass
class SegmentNode:
    name: str
    level: int
    children: list[Self] = field(default_factory=list)
    content: str = ""

    def add_content(self, data: str) -> None:
        if not self.content:
            self.content = data
        else:
            self.content += "\n" + data


@dataclass
class VectorizedSegmentNode:
    name: str
    level: int
    embedding: np.ndarray | None
    content: list[tuple[str, np.ndarray]]
    children: list[Self] = field(default_factory=list)
