from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from cwm.testing import build_tiny_model

    return build_tiny_model(tmp_path_factory.mktemp("tiny-model"))


@pytest.fixture(scope="session")
def tiny_model(tiny_model_dir: Path):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(tiny_model_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(tiny_model_dir)
    return model, tokenizer


@pytest.fixture()
def mini_dataset() -> Path:
    return FIXTURES / "mini_problems.jsonl"
