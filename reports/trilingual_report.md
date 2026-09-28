# Trilingual consistency report (English ↔ Hindi ↔ Hinglish)

Generated 2026-09-26 17:08 · mode **retrieval-only** · 14/14 items evaluated

## Thresholds

Evidence overlap ≥ 0.6 · missing evidence = 0 · plan consistency ≥ 0.9 · answer similarity ≥ 0.85 · hallucination rate ≤ 0.2

## Summary

```
Mode: retrieval-only   Items: 14/14 evaluated
Hinglish detected: 14/14   English misdetected as Hinglish: 0/14
Retrieval Hindi                   : overlap 0.94, passed 14/14, missing evidence 2, same canonical 8/14
Retrieval Hinglish (normalised)   : overlap 1.00, passed 14/14, missing evidence 0, same canonical 10/14
Retrieval Hinglish, old pipeline  : overlap 0.97, passed 14/14, missing evidence 1, same canonical 7/14
Retrieval Hinglish, raw embedding : overlap 0.21, passed 4/14, missing evidence 42, same canonical 0/14
Spelling variants → same canonical question: 8/14
```

## Retrieval: normaliser on vs off

| Retrieval of the Hinglish question | Mean overlap vs English | Passed | Missing evidence | Same canonical |
|---|---|---|---|---|
| Hindi | 0.94 | 14/14 | 2 | 8/14 |
| Hinglish (normalised) | 1.00 | 14/14 | 0 | 10/14 |
| Hinglish, old pipeline | 0.97 | 14/14 | 1 | 7/14 |
| Hinglish, raw embedding | 0.21 | 4/14 | 42 | 0/14 |

## Per item: retrieval

| ID | Safety | HI overlap | HG overlap | HG missing | HG raw overlap | Same canonical HI/HG | Variants same |
|---|---|---|---|---|---|---|---|
| T1 | ✗ | 1.00 | 1.00 | 0 | 0.00 | ✓/✓ | 3/3 |
| T2 | ✓ | 1.00 | 1.00 | 0 | 0.00 | ✓/✓ | 1/1 |
| T3 | ✗ | 0.60 | 1.00 | 0 | 0.00 | ✗/✓ | 1/2 |
| T4 | ✓ | 1.00 | 1.00 | 0 | 0.00 | ✓/✓ | 1/1 |
| T5 | ✓ | 1.00 | 1.00 | 0 | 0.00 | ✓/✓ | 0/0 |
| T6 | ✗ | 1.00 | 1.00 | 0 | 1.00 | ✗/✗ | 0/1 |
| T7 | ✗ | 1.00 | 1.00 | 0 | 0.60 | ✗/✓ | 0/0 |
| T8 | ✗ | 1.00 | 1.00 | 0 | 0.00 | ✗/✗ | 0/1 |
| T9 | ✗ | 1.00 | 1.00 | 0 | 0.60 | ✗/✗ | 0/1 |
| T10 | ✗ | 1.00 | 1.00 | 0 | 0.00 | ✓/✓ | 1/2 |
| T11 | ✗ | 1.00 | 1.00 | 0 | 0.60 | ✓/✓ | 0/0 |
| T12 | ✗ | 1.00 | 1.00 | 0 | 0.00 | ✓/✓ | 1/1 |
| T13 | ✗ | 0.60 | 1.00 | 0 | 0.14 | ✗/✗ | 0/1 |
| T14 | ✓ | 1.00 | 1.00 | 0 | 0.00 | ✓/✓ | 0/0 |

## Canonical-question mismatches

Questions that did not converge on one canonical question (each one is a potential answer drift):

- **T3**
  - EN: Should I have a snack before exercising?
  - HI: Should I eat before exercising?
  - HG: Should I have a snack before exercising?
- **T6**
  - EN: What are the BMI cut-offs for obesity in Indian adults?
  - HI: What is the BMI range for obesity in Indian adults?
  - HG: What is the BMI cut‑off for obesity in Indian adults?
- **T7**
  - EN: What is the recommended waist circumference for an Indian adult?
  - HI: What should my waist circumference be for an Indian adult?
  - HG: What is the recommended waist circumference for an Indian adult?
- **T8**
  - EN: Why should I track HbA1c instead of just my weight?
  - HI: Why should I monitor HbA1c instead of just weight?
  - HG: Why should I track HbA1c instead of just weight?
- **T9**
  - EN: Can type 2 diabetes go into remission with weight loss?
  - HI: Can weight loss lead to remission of type 2 diabetes?
  - HG: Can weight loss lead to type 2 diabetes remission?
- **T13**
  - EN: How much should I walk each day to lose weight?
  - HI: How much should I walk daily to lose weight?
  - HG: How much walking should I do daily for weight loss?
