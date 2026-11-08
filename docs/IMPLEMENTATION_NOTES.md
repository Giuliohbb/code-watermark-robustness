# Implementation notes

Design decisions that affect the numbers, and the places where the reference
implementations differ from the papers.

## One green list for every method

KGW, SWEET, STONE and the EWD detector share a single `GreenList`
(`src/cwm/watermarks/greenlist.py`). Methods differ only in *where* they add the bias
and *which tokens they score*, so a difference in robustness cannot come from the
hashing itself.

The partition follows MarkLLM's `time` scheme with a context of one token, which is
also what the official STONE code uses:

```
seed  = hash_key * (prev_token % vocab_size)
green = randperm(vocab_size, seed)[: int(gamma * vocab_size)]
```

The generator always runs on **CPU**. `torch.randperm` returns different permutations
on CPU and CUDA for the same seed, so seeding on the model device would make GPU
generation and CPU detection disagree. `vocab_size` is the size of the model's output
layer (`model.config.vocab_size`, 152,064 for Qwen2.5-Coder-7B), not `len(tokenizer)`.

## Order of logits processors

Sampling settings are passed through a fresh `GenerationConfig`, so nothing is
inherited from the model's defaults (Qwen ships with top-k 20, top-p 0.8, temperature
0.7 and a repetition penalty). The watermark runs on the **raw logits** and top-k runs
after it:

```
[temperature (only if != 1)] -> watermark -> top-k -> top-p -> sample
```

In MarkLLM-based code, top-k runs first, so SWEET's entropy test and STONE's candidate
test see a truncated distribution at generation time but the full distribution at
detection time. Running them on the raw logits removes that mismatch.

## Detection sees text, with the prompt as context

Detectors re-tokenize the code text, because that is all a real detector gets. If the
model sampled a non-canonical tokenization (`"de" + "f"` instead of `"def"`), the
detector scores different tokens. The report measures this as
`retokenization_agreement` in the generation table.

The chat prompt, which ends at the opening ` ```python ` fence, is used as context. It
seeds the green list of the first code token, and it lets SWEET and EWD recompute the
entropy the model actually had (the main setting of the SWEET paper). Human reference
solutions get the same context. Set `detection.use_prompt_context: false` to score
without it, as MarkLLM does.

## STONE: paper vs. official code

We read `watermark/stone/stone.py` in
[inistory/STONE-watermarking](https://github.com/inistory/STONE-watermarking) (commit
`bb5d809`) and found three differences from the paper. Each one is configurable.

| Aspect | Paper | Official code | Our default | Option |
| --- | --- | --- | --- | --- |
| Candidate token that decides whether to watermark | sampled (Algorithm 1) | argmax of the raw distribution | argmax | `stone_candidate` |
| z-score denominator | number of non-syntax tokens (Algorithm 2) | number of **syntax** tokens | paper | `stone_denominator` |
| Syntax test for a model token | Table 9 elements | decoded text, or text stripped of spaces, is in Table 9 | official | `stone_syntax_rule` |

**Candidate.** The code takes `argmax(softmax(scores))` instead of sampling a candidate.
We keep the official behaviour because it produced the published numbers.

**Denominator.** In `score_sequence`, `target_num` counts the *non*-syntax tokens and
`num_tokens_scored = len(input_ids) - prefix_length - target_num`, so the z-score
divides a count over non-syntax tokens by a total over syntax tokens. We default to the
paper's formula and keep `official` to reproduce the released code.

**Syntax test.** Because a token is syntactic only if its text (or stripped text) is
exactly a listed element, merged BPE tokens such as `"):"` and pure indentation such as
`"    "` are *not* protected and get watermarked. The `strict` rule also protects
whitespace-only tokens and tokens made only of operator or delimiter characters.

## SWEET and EWD

SWEET uses Shannon entropy with threshold 0.9, the optimum reported in Appendix E of
STONE. EWD weights each token by its spike entropy minus the sample's minimum, with
modulus `m = (1-γ)(e^δ-1) / (1-γ+e^δ·γ)`, and uses the weighted z-score of the official
code. EWD only changes detection, so it re-scores the KGW generations.

## Thresholds and negatives

For each detector the threshold is set on the **untransformed** negatives at a target
false positive rate and then kept fixed. After a transformation we report both the TPR
on watermarked code and the FPR on transformed negatives, so a transformation that
inflates the scores of human code is also visible. Negatives are the human reference
solutions (`prompt + canonical_solution`) and the unwatermarked generations.

## Functional correctness

`evalplus.evaluate` refuses sample files that do not cover every task, which rules out
the 20-task pilot. `cwm.evaluation.functional` calls EvalPlus's own
`check_correctness` per sample, with expected outputs from `get_groundtruth`. A sample
passes when both the base and the plus tests pass. Every transformation is re-tested;
`outcome_changes` reports how often a transformation changed a test outcome, which
should be zero.
