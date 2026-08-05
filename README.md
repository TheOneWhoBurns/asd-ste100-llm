# ASD-STE100 language post-training lab

The goal is narrow: make a local language model produce English in the
ASD-STE100 style. This project does not target aviation-manual publication.

The project now includes a deterministic reward harness. It checks model output
and returns a bounded scalar reward plus machine-readable findings. The checker
engine and its language model are pinned in `uv.lock`.

## Install and test

```sh
make env
make reward-test
make reward-check
make reward-benchmark
make chat
```

The test suite covers normal rules and reward-gaming attempts.
The environment setup also materializes its interpreter as ordinary files.

Once the model checkpoint is present, `make chat` starts a local interactive
conversation. Each answer is checked and can be regenerated up to two times
with the checker findings included as repair instructions. Use `/quit` to exit.
The training reward remains strict; interactive chat additionally permits
open-class technical terms introduced by the current answer so a general
conversation is not forced to delete its subject vocabulary.

## Check one output

```sh
printf '%s\n' 'The server is hot.' |
  .venv/bin/ste-reward check \
    --text-type description \
    --glossary policy/glossary.yaml \
    --term server \
    --pretty
```

The command writes JSON. It exits with status 0 when the strict gate passes and
status 1 when it fails. Use `--no-strict` if warnings must not fail the gate.

The output includes:

- `reward`: scalar in `[-1, 1]`
- `compliant`: no error findings
- `strict_compliant`: no error or warning findings
- `rule_counts`, `summary`, and span-level `findings`
- the pinned checker revision

## Score JSONL

Each input row must contain an `output` string. It can also contain a
`text_type` value of `procedure`, `description`, or `auto`, and a
`technical_terms` array for terms that are valid in that task.

```json
{"output":"The server is hot.","text_type":"description","technical_terms":["server"]}
```

```sh
.venv/bin/ste-reward batch \
  --input generations.jsonl \
  --output scored.jsonl \
  --glossary policy/glossary.yaml
```

The scorer preserves all input fields and adds a `ste_reward` object.

## Use it as an RL reward function

```python
from ste_reward import STERewardFunction

reward_function = STERewardFunction(
    text_type="auto",
    glossary_path="policy/glossary.yaml",
)

rewards = reward_function(
    completions,
    text_type=["procedure", "description"],
    technical_terms=[["router"], ["language model", "training data"]],
)
```

`completions` can contain raw strings, chat message lists, or mappings with a
`content` or `text` field. The callable returns one float per completion and
accepts per-row `text_type` and `technical_terms` columns, which makes it
suitable for batch RL trainers.

Strict compliance is lexicographic:

- a clean output gets a positive reward, normally `1.0`;
- any error or warning makes the default strict reward non-positive;
- empty, punctuation-only, numeric-only, non-English, and all-caps bypass
  outputs receive negative rewards;
- more or heavier findings move the result toward `-1.0`.

Combine this style reward with a task or meaning reward. If the trainer
optimizes only the style reward, a very short valid sentence can beat a useful
answer.

## Machine-checked policy

The harness checks the controlled vocabulary and project terms, the 20-word
procedure limit, the 25-word description limit, imperative procedure steps,
passive constructions, verb forms, long noun clusters, multiple instructions,
ambiguous pronouns, weak description openings, semicolons, contractions, and
the six-sentence description-paragraph limit.

`policy/glossary.yaml` extends the controlled vocabulary for this model. Enter
each token of a multiword technical term because vocabulary is evaluated one
token at a time. For mixed-domain training, prefer the per-example
`technical_terms` field. Its terms suppress only unapproved-vocabulary
findings; they do not suppress forbidden constructions or grammar findings.

The checker is deterministic, but some language rules depend on syntax and
meaning. Its findings are therefore a useful training signal, not an oracle.
Frozen adversarial and human preference evaluations should remain separate
from the reward used for training.

## Local model

The pinned backbone is `mlx-community/Qwen3.5-9B-MLX-4bit` at commit
`938d8919941c6e7efd3c7150eff7fe9d12afa631`. It is a 4-bit MLX conversion of
`Qwen/Qwen3.5-9B`; LoRA training keeps the backbone frozen.

```sh
make model
make verify-model
make standard
make baseline
make trainability-probe
```

The model weights, reference standard, private corpora, processed examples, and
adapters are excluded from Git.

Do not train from the standard alone. Build pairs of source text and accepted
rewrites, label each pair as procedural or descriptive, and split by source
document before augmentation. Keep rewrites from the same source out of both
training and evaluation.
