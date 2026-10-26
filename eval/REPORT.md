# Evaluation report

Corpus: 554 chunks from 2 documents; dense embedder: `tfidf`.
37 labelled questions (hand-written; label pattern verified to exist in source) + 20 out-of-scope questions.

## Retrieval (hit = a top-k chunk from the right document contains the answer text)

| method | hit@1 | hit@3 | hit@5 | MRR | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| bm25 | 0.73 | 0.784 | 0.838 | 0.776 | 1.9 | 4.5 |
| dense (tf-idf) | 0.514 | 0.73 | 0.757 | 0.619 | 1.7 | 3.9 |
| hybrid (RRF) | 0.676 | 0.757 | 0.838 | 0.735 | 1.7 | 3.4 |
| hybrid + lexical rerank | 0.622 | 0.757 | 0.784 | 0.706 | 2.7 | 4.8 |

## Answers / abstention

- answer_contains_fact: 0.622
- false_abstain_rate: 0.162
- correct_abstain_rate_out_of_scope: 0.9
- note: extractive answers (no LLM): 'answer_contains_fact' is a strict, lower-bound measure

## Text-to-SQL vs pandas ground truth: 10/10 checks passed

## Security

- sql_attacks_blocked: 25/25
- injection_patterns_detected: 5/5

## Limits
- Questions are few (37) and hand-written by the project author; widen before drawing strong conclusions.
- Default dense retriever is TF-IDF, so 'dense' here is lexical. Re-run with EMBEDDER=st/fastembed for real semantic embeddings.
- The abstention threshold (COVERAGE_MIN) was chosen by looking at these same questions, so abstention numbers are optimistic.
- Answers are extractive (no LLM) unless ANTHROPIC_API_KEY is set; faithfulness with an LLM is NOT measured here.