from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class FixedChunking:
    chunk_size: int
    overlap: int

    def split(self, text: str) -> List[str]:
        if not text:
            return []
        step = max(1, self.chunk_size - self.overlap)
        chunks = []
        start = 0
        while start < len(text):
            chunk = text[start : start + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk)
            start += step
        return chunks

