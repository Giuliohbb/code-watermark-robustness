"""Command-line interface: one command per pipeline stage, plus ``run`` for all of them.

Example::

    cwm run configs/pilot.yaml
    cwm detect configs/pilot.yaml --set detectors='[kgw, stone]'
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml

from cwm.config import ExperimentConfig, load_config, save_config

app = typer.Typer(add_completion=False, help="Robustness of LLM code watermarks to code edits.")

ConfigArg = Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="YAML config")]
SetOpt = Annotated[
    list[str] | None,
    typer.Option(
        "--set", help="Override a config key, e.g. task_limit=5 or generation.n_samples=2"
    ),
]


def _overrides(pairs: list[str] | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for pair in pairs or []:
        key, _, raw = pair.partition("=")
        if not key or not _:
            raise typer.BadParameter(f"Expected key=value, got {pair!r}")
        value = yaml.safe_load(raw)
        node = result
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value
    return result


def _load(config_path: Path, sets: list[str] | None) -> ExperimentConfig:
    config = load_config(config_path, _overrides(sets))
    config.run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, config.run_dir / "config.yaml")
    return config


def _needs_model_for_detection(config: ExperimentConfig) -> bool:
    return any(d in ("sweet", "ewd") for d in config.detectors)


def _stage(name: str) -> float:
    typer.secho(f"\n== {name} ==", fg=typer.colors.CYAN, bold=True)
    return time.perf_counter()


def _done(start: float) -> None:
    typer.echo(f"done in {time.perf_counter() - start:.1f}s")


@app.command()
def generate(config: ConfigArg, set_: SetOpt = None) -> None:
    """Sample code for every task and watermarking condition."""
    from cwm.generation import run_generation
    from cwm.models import load_model

    cfg = _load(config, set_)
    start = _stage("generate")
    model, tokenizer = load_model(cfg.model)
    run_generation(cfg, model, tokenizer)
    _done(start)


@app.command()
def transform(config: ConfigArg, set_: SetOpt = None) -> None:
    """Apply every transformation to generated and human code."""
    from cwm.transforms import run_transforms

    cfg = _load(config, set_)
    start = _stage("transform")
    run_transforms(cfg)
    _done(start)


@app.command()
def functional(config: ConfigArg, set_: SetOpt = None) -> None:
    """Run the EvalPlus tests on every variant."""
    from cwm.evaluation.functional import run_functional

    cfg = _load(config, set_)
    start = _stage("functional")
    run_functional(cfg)
    _done(start)


@app.command()
def detect(config: ConfigArg, set_: SetOpt = None) -> None:
    """Score every variant with every detector."""
    from cwm.detection import run_detection
    from cwm.models import load_model, load_tokenizer, output_vocab_size

    cfg = _load(config, set_)
    start = _stage("detect")
    if _needs_model_for_detection(cfg):
        cfg.model.device = cfg.detection.device
        model, tokenizer = load_model(cfg.model)
        vocab_size = int(model.config.vocab_size)
    else:
        model, tokenizer, vocab_size = None, load_tokenizer(cfg.model), output_vocab_size(cfg.model)
    run_detection(cfg, tokenizer, vocab_size, model=model)
    _done(start)


@app.command()
def report(config: ConfigArg, set_: SetOpt = None) -> None:
    """Aggregate metrics into CSV tables and a Markdown report."""
    from cwm.evaluation.report import run_report
    from cwm.models import load_tokenizer

    cfg = _load(config, set_)
    start = _stage("report")
    path = run_report(cfg, tokenizer=load_tokenizer(cfg.model))
    typer.echo(f"report written to {path}")
    _done(start)


@app.command()
def run(config: ConfigArg, set_: SetOpt = None) -> None:
    """All stages in order, loading the model once."""
    from cwm.detection import run_detection
    from cwm.evaluation.functional import run_functional
    from cwm.evaluation.report import run_report
    from cwm.generation import run_generation
    from cwm.models import load_model
    from cwm.transforms import run_transforms

    cfg = _load(config, set_)
    model, tokenizer = load_model(cfg.model)

    start = _stage("generate")
    run_generation(cfg, model, tokenizer)
    _done(start)
    start = _stage("transform")
    run_transforms(cfg)
    _done(start)
    start = _stage("functional")
    run_functional(cfg)
    _done(start)
    start = _stage("detect")
    run_detection(cfg, tokenizer, int(model.config.vocab_size), model=model)
    _done(start)
    start = _stage("report")
    path = run_report(cfg, tokenizer=tokenizer)
    typer.echo(f"report written to {path}")
    _done(start)


if __name__ == "__main__":  # pragma: no cover
    app()
