# Methodology

## Research questions

1. Under one protocol, how much detectability do KGW, SWEET and STONE lose under each
   family of semantics-preserving transformation, and is routine maintenance alone
   enough to remove them?
2. Which lexical categories of code carry the watermark signal, and does that explain
   which transformations hurt each method the most?

## Conditions

| Condition | Generation | Detectors |
| --- | --- | --- |
| unwatermarked | plain sampling | used as negatives |
| kgw | bias on every token | `kgw`, `ewd` |
| sweet | bias where entropy > 0.9 | `sweet` |
| stone | bias where the next token is not syntactic | `stone` |

All methods share γ = 0.5, δ = 1.0 and the same hash key, so robustness differences
do not come from different watermark strengths.

## Transformations

| Family | Transform | Tool |
| --- | --- | --- |
| maintenance | `black` | Black, default style |
| maintenance | `strip_comments` | LibCST: comments and docstrings removed, layout kept |
| attack | `rename_0.5`, `rename_1.0` | LibCST scope analysis: local variables renamed to realistic names; parameters, functions and imports untouched |
| attack | `dead_code_3` | three never-executed or no-op statements, as in Suresh et al. (2024) |
| attack | `minify` | python-minifier, as in Disappearing Ink (Zhang et al., 2025) |

Every transformed program is re-run against the EvalPlus tests to confirm that its
behaviour did not change.

## Metrics

- **pass@1**: mean over tasks of the fraction of passing samples (base and plus tests).
- **AUROC**: separation between watermarked and negative z-scores, both transformed.
- **TPR / FPR at 1% and 5%**: threshold chosen on untransformed negatives, then fixed.
- **z retention**: mean z after the transformation divided by mean z before, over the
  same watermarked samples.
- **Signal decomposition**: the z-score of count-based detectors is a sum over tokens,
  so it splits exactly into one contribution per lexical category (identifier,
  keyword, operator, delimiter, literal, docstring, comment, whitespace).

## Setup

- Model: `Qwen/Qwen2.5-Coder-7B-Instruct`, bf16, chat template with the EvalPlus
  instruction and a response pre-filled up to the opening code fence.
- Sampling: temperature 1.0, top-k 50, top-p 1.0, up to 1,024 new tokens (STONE's
  sampling; STONE used the base model, so this is not an exact reproduction).
- Pilot: 20 HumanEval+ tasks × 3 samples × 4 conditions = 240 solutions.
- Final: 164 HumanEval+ tasks, number of samples per task set from the pilot's cost.

## Out of scope

Paraphrasing with an LLM, trained watermarks (SWaRL), LLM-applied transformations
(CLASP), model watermarks (CodeGenGuard), languages other than Python, and the
STONE variant that is invariant to formatting.
