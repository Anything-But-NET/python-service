from dataclasses import dataclass, field
from typing import Self

import numpy as np
import uuid
from typing import Any
from datetime import datetime


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

@dataclass
class PromptVersion:
    id: uuid.UUID
    prompt_text: str
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class EvaluationRun:
    id: uuid.UUID
    prompt_version_id: uuid.UUID
    status: str
    metadata: dict = field(default_factory=dict)
    created_at: datetime | None = None
