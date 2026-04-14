from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import json


@dataclass
class Segment:
    label: str
    start: float
    end: float
    duration: float
    score: float
    source: str


@dataclass
class Instance:
    label: str
    start: float
    end: float
    duration: float
    rep_id: int
    score: float


@dataclass
class InferenceSchema:
    video_path: str
    segments: list[Segment]
    instances: list[Instance]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_path": self.video_path,
            "segments": [asdict(x) for x in self.segments],
            "instances": [asdict(x) for x in self.instances],
            "summary": self.summary,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
