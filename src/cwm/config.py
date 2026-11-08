"""Experiment configuration, loaded from YAML into typed dataclasses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

import yaml

Dataset = Literal["humaneval", "mbpp"]

CONDITIONS = ("unwatermarked", "kgw", "sweet", "stone")
DETECTORS = ("kgw", "ewd", "sweet", "stone")
# Generation condition whose samples each detector is meant to find.
DETECTOR_SOURCE = {"kgw": "kgw", "ewd": "kgw", "sweet": "sweet", "stone": "stone"}


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    revision: str | None = None
    dtype: str = "bfloat16"
    device: str = "cuda"


@dataclass
class GenerationConfig:
    n_samples: int = 3
    max_new_tokens: int = 1024
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    seed: int = 1234


@dataclass
class WatermarkConfig:
    gamma: float = 0.5
    delta: float = 1.0
    hash_key: int = 15485863
    sweet_entropy_threshold: float = 0.9
    stone_syntax_rule: Literal["official", "strict"] = "official"
    stone_candidate: Literal["argmax", "sample"] = "argmax"
    stone_denominator: Literal["paper", "official"] = "paper"
    greenlist_cache_size: int = 1024


@dataclass
class TransformSpec:
    name: str
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        """Stable identifier, e.g. ``rename_1.0``."""
        if not self.params:
            return self.name
        return self.name + "_" + "_".join(str(v) for _, v in sorted(self.params.items()))


@dataclass
class DetectionConfig:
    device: str = "cuda"
    use_prompt_context: bool = True
    save_token_details: bool = True


@dataclass
class EvaluationConfig:
    fpr_targets: list[float] = field(default_factory=lambda: [0.01, 0.05])
    negatives: list[str] = field(default_factory=lambda: ["human", "unwatermarked"])
    functional_workers: int = 4
    bootstrap_samples: int = 0


@dataclass
class ExperimentConfig:
    run_name: str = "pilot"
    output_dir: str = "runs"
    dataset: Dataset = "humaneval"
    # Optional local JSONL in EvalPlus format, used instead of downloading the dataset.
    dataset_path: str | None = None
    task_limit: int | None = 20
    task_selection_seed: int = 0
    conditions: list[str] = field(default_factory=lambda: list(CONDITIONS))
    detectors: list[str] = field(default_factory=lambda: list(DETECTORS))
    transforms: list[TransformSpec] = field(default_factory=list)
    transform_seed: int = 7
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    watermark: WatermarkConfig = field(default_factory=WatermarkConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.run_name

    def validate(self) -> None:
        unknown = set(self.conditions) - set(CONDITIONS)
        if unknown:
            raise ValueError(f"Unknown conditions: {sorted(unknown)}")
        unknown = set(self.detectors) - set(DETECTORS)
        if unknown:
            raise ValueError(f"Unknown detectors: {sorted(unknown)}")
        missing = {DETECTOR_SOURCE[d] for d in self.detectors} - set(self.conditions)
        if missing:
            raise ValueError(f"Detectors need these generation conditions: {sorted(missing)}")
        if "unwatermarked" in self.evaluation.negatives and "unwatermarked" not in self.conditions:
            raise ValueError("The 'unwatermarked' negative set requires that condition")
        if self.dataset not in ("humaneval", "mbpp"):
            raise ValueError(f"Unknown dataset: {self.dataset}")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["transforms"] = [{"name": t.name, **t.params} for t in self.transforms]
        return data


def _build(cls: type, data: dict[str, Any] | None) -> Any:
    data = data or {}
    known = {f.name for f in fields(cls)}
    extra = set(data) - known
    if extra:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(extra)}")
    return cls(**data)


def config_from_dict(raw: dict[str, Any]) -> ExperimentConfig:
    """Build and validate a config from a plain dictionary (e.g. parsed YAML)."""
    raw = dict(raw)
    transforms = [
        TransformSpec(name=item["name"], params={k: v for k, v in item.items() if k != "name"})
        for item in raw.pop("transforms", []) or []
    ]
    sections = {
        "model": ModelConfig,
        "generation": GenerationConfig,
        "watermark": WatermarkConfig,
        "detection": DetectionConfig,
        "evaluation": EvaluationConfig,
    }
    nested = {key: _build(cls, raw.pop(key, None)) for key, cls in sections.items()}
    config = _build(ExperimentConfig, raw)
    for key, value in nested.items():
        setattr(config, key, value)
    config.transforms = transforms
    config.validate()
    return config


def merge_overrides(raw: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overrides`` into a copy of ``raw``."""
    merged = dict(raw)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_overrides(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    """Load a YAML config, optionally merging nested ``overrides`` on top."""
    with open(path, encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}
    return config_from_dict(merge_overrides(raw, overrides or {}))


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False, allow_unicode=True)
