from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import yaml


def _slug(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


@dataclass
class RuntimeSelection:
    model: str
    language: str


class DistributedConfig:
    def __init__(self, path: str):
        self.path = Path(path)
        with self.path.open("r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

    def resolve_selection(self, cli_model: str | None, cli_lang: str | None) -> RuntimeSelection:
        model = cli_model or self.cfg["run"]["model"]
        language = (cli_lang or self.cfg["run"]["language"]).strip().lower()
        if language not in {"ru", "kg", "en"}:
            raise ValueError("language must be one of: ru, kg, en")
        return RuntimeSelection(model=model, language=language)

    def model_config(self, model_name: str) -> Dict:
        registry = self.cfg["models_registry"]
        if model_name not in registry:
            known = ", ".join(sorted(registry.keys()))
            raise ValueError(f"Model '{model_name}' not found. Available: {known}")
        cfg = dict(registry[model_name])
        cfg["name"] = model_name
        return cfg

    def collection_name(self, dataset_name: str, model_name: str) -> str:
        chunk_size = int(self.cfg["frozen"]["chunking"]["chunk_size"])
        overlap = int(self.cfg["frozen"]["chunking"]["overlap"])
        return f"{_slug(dataset_name)}_{_slug(model_name)}_chunk{chunk_size}_overlap{overlap}"

