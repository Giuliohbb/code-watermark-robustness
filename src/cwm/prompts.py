"""Chat prompts in the EvalPlus format for instruction-tuned models.

The assistant turn is pre-filled up to an opening ```python fence, as EvalPlus does
(``evalplus.provider.utility.make_raw_chat_prompt``). The model's response is then
code from its first token, and the code ends at the closing fence.
"""

from __future__ import annotations

from typing import Any

INSTRUCTION_PREFIX = (
    "Please provide a self-contained Python script that solves the following problem "
    "in a markdown code block:"
)
RESPONSE_PREFIX = (
    "Below is a Python script with a self-contained function that solves the problem "
    "and passes corresponding tests:"
)
CODE_FENCE = "```"
_SPLITTER = "-[[]]-cwm-response-splitter-[[]]-"


def build_chat_prompt(task_prompt: str, tokenizer: Any) -> str:
    if getattr(tokenizer, "chat_template", None) is None:
        raise ValueError("The tokenizer has no chat template; use an instruction-tuned model")
    user = f"{INSTRUCTION_PREFIX}\n```\n{task_prompt.strip()}\n```\n"
    assistant = f"{RESPONSE_PREFIX}\n```python\n{_SPLITTER}\n```\n"
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}],
        tokenize=False,
    )
    return rendered.split(_SPLITTER)[0]


def extract_code(response: str) -> tuple[str, bool]:
    """Code before the closing fence, and whether the fence was found."""
    end = response.find(CODE_FENCE)
    if end == -1:
        return response, False
    return response[:end], True
