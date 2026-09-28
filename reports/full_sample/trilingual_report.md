# Trilingual consistency report (English ↔ Hindi ↔ Hinglish)

Generated 2026-09-26 17:14 · mode **full** · 3/3 items evaluated

## Thresholds

Evidence overlap ≥ 0.6 · missing evidence = 0 · plan consistency ≥ 0.9 · answer similarity ≥ 0.85 · hallucination rate ≤ 0.2

## Summary

```
Mode: full   Items: 3/3 evaluated
Hinglish detected: 3/3   English misdetected as Hinglish: 0/3
Retrieval Hindi                   : overlap 1.00, passed 3/3, missing evidence 0, same canonical 2/3
Retrieval Hinglish (normalised)   : overlap 1.00, passed 3/3, missing evidence 0, same canonical 2/3
Retrieval Hinglish, raw embedding : overlap 0.20, passed 1/3, missing evidence 9, same canonical 0/3
Spelling variants → same canonical question: 4/5
Hindi     plan 0.89 (1/3), answer sim 0.93 (3/3), hallucination 0.09, parity 3/3, same answer 2/3
Hinglish  plan 0.92 (2/3), answer sim 0.97 (3/3), hallucination 0.21, parity 3/3, same answer 2/3
```

## Retrieval: normaliser on vs off

| Retrieval of the Hinglish question | Mean overlap vs English | Passed | Missing evidence | Same canonical |
|---|---|---|---|---|
| Hindi | 1.00 | 3/3 | 0 | 2/3 |
| Hinglish (normalised) | 1.00 | 3/3 | 0 | 2/3 |
| Hinglish, raw embedding | 0.20 | 1/3 | 9 | 0/3 |

## Per item: retrieval

| ID | Safety | HI overlap | HG overlap | HG missing | HG raw overlap | Same canonical HI/HG | Variants same |
|---|---|---|---|---|---|---|---|
| T1 | ✗ | 1.00 | 1.00 | 0 | 0.00 | ✓/✓ | 3/3 |
| T2 | ✓ | 1.00 | 1.00 | 0 | 0.00 | ✓/✓ | 1/1 |
| T9 | ✗ | 1.00 | 1.00 | 0 | 0.60 | ✗/✗ | 0/1 |

## Per item: plan and answer

| ID | Lang | Same answer | Plan | Answer sim | Hallucination | Parity | Failed checks |
|---|---|---|---|---|---|---|---|
| T1 | Hindi | ✓ | 1.00 | 0.99 | 0.00 | ✓ | — |
| T1 | Hinglish | ✓ | 1.00 | 1.00 | 0.40 | ✓ | — |
| T2 | Hindi | ✓ | 0.81 | 0.92 | 0.10 | ✓ | — |
| T2 | Hinglish | ✓ | 0.91 | 0.96 | 0.09 | ✓ | — |
| T9 | Hindi | ✗ | 0.88 | 0.89 | 0.17 | ✓ | — |
| T9 | Hinglish | ✗ | 0.83 | 0.95 | 0.14 | ✓ | — |

## Canonical-question mismatches

Questions that did not converge on one canonical question (each one is a potential answer drift):

- **T9**
  - EN: Can type 2 diabetes go into remission with weight loss?
  - HI: Can weight loss lead to remission of type 2 diabetes?
  - HG: Can weight loss lead to type 2 diabetes remission?
