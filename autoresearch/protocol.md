# Autoresearch Protocol: STE reward harness

## Objective

Improve strict acceptance of useful GPT-generated STE-style outputs while
preserving rejection of known reward-gaming and rule-violating outputs.

## Editable Surface

- `src/ste_reward/scorer.py`
- `src/ste_reward/integrations.py`
- `src/ste_reward/cli.py`
- `policy/glossary.yaml`
- tests and evaluation fixtures only when adding a new confirmed failure mode

## Evaluation

- Command: `make reward-autoresearch`
- Primary metric: positive-set strict pass rate
- Direction: maximize
- Secondary metrics: adversarial rejection rate, runtime, full regression tests
- Budget: 6 iterations, maximum 5 minutes per evaluation

## Promotion Rule

Keep only when the positive strict pass rate improves and adversarial rejection
remains 100%. A tie is not a keeper unless it reduces runtime or complexity.
Every candidate must also pass `make reward-test`.

## Validation

- Baseline: `data/eval/gpt5_positive.jsonl` (12 generated outputs)
- Training/easy set: first 8 positive rows during a targeted experiment
- Validation set: `data/eval/command_validation.jsonl` (7 imperative commands)
- Adversarial/counterexample set: `data/eval/adversarial.jsonl` (8 rows)

## Notes

- Keep failed attempts in `ledger.jsonl`.
- Do not treat empirical wins as proofs.
