# spark-benchmark report

- runs root: results/runs
- total runs: 14

## hallucination_grounding_v1

| model | passes | total | pass_rate | runs |
| --- | ---: | ---: | ---: | ---: |
| nemotron-3 | 19 | 24 | 79.17% | 6 |
| gemma-4 | 18 | 24 | 75.00% | 6 |
| qwen-3.6 | 15 | 24 | 62.50% | 6 |

## practical_structured_output_v1

| model | passes | total | pass_rate | runs |
| --- | ---: | ---: | ---: | ---: |
| gemma-4 | 11 | 12 | 91.67% | 2 |
| qwen-3.6 | 11 | 12 | 91.67% | 2 |
| nemotron-3 | 10 | 12 | 83.33% | 2 |

## Recent Runs

| run_id | experiment | backend | suite | rows |
| --- | --- | --- | --- | ---: |
| 20260519T065915Z-c683a252 | spark-ollama-v1-baseline | ollama | practical_structured_output_v1 | 18 |
| 20260519T065801Z-e53c4982 | spark-ollama-v1-baseline | ollama | practical_structured_output_v1 | 18 |
| 20260519T065154Z-24f1b3e4 | spark-ollama-v1-baseline | ollama | hallucination_grounding_v1 | 27 |
| 20260518T211724Z-ad515cd0 | spark-ollama-v1-baseline | ollama | hallucination_grounding_v1 | 9 |
| 20260518T211536Z-06443d6e | spark-ollama-v1-baseline | ollama | - | 3 |
| 20260518T191109Z-a2974a13 | spark-ollama-v1-baseline | ollama | hallucination_grounding_v1 | 9 |
| 20260518T191000Z-e0204fc2 | spark-ollama-v1-baseline | ollama | - | 8 |
| 20260518T190855Z-e3999da8 | spark-ollama-v1-baseline | ollama | hallucination_grounding_v1 | 9 |
| 20260518T190806Z-071f70ea | spark-ollama-v1-baseline | ollama | hallucination_grounding_v1 | 9 |
| 20260518T190730Z-c2a4674a | spark-ollama-v1-baseline | ollama | hallucination_grounding_v1 | 9 |

