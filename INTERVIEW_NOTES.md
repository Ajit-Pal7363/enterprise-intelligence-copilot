# Interview notes (read the code first, then rehearse these)

**30-second pitch.** Natural-language interface over three sources: PDFs (RAG with citations and abstention),
a 99k-order SQL database (validated, read-only Text-to-SQL), and customer reviews. A router sends each question
to the right engine; everything is evaluated against labelled questions and pandas ground truth.

**Decisions you must be able to defend**
1. *Why BM25 + dense + RRF?* Keyword search wins on exact terms ("forty-eight hours"), dense wins on paraphrase.
   RRF fuses ranks without calibrating scores. **Measured result:** hybrid ≈ BM25 here (hit@3 0.757 both) because my
   "dense" is TF-IDF. Say this plainly; it shows you measure instead of assume.
2. *Why did you remove the reranker?* The lexical reranker lowered hit@1 (0.65 → 0.60). A cross-encoder is the next
   experiment (`RERANKER=ce`), not a claim.
3. *Why templates for SQL, not an LLM?* Reliability and testability: 10/10 checks equal pandas ground truth.
   The LLM is a fallback that still passes the same validator.
4. *Why three SQL safety layers?* Regex validation is bypassable (I have a test proving a comma-join slips through);
   the SQLite authorizer enforces at engine level; read-only mode + timeout + row cap limit blast radius.
5. *ACL before ranking.* Filtering after retrieval can leak through scores/ranks and wastes top-k slots.
6. *Abstention.* An IDF-weighted out-of-vocabulary gate. Trade-off measured: stricter = more false abstains.
   Tuned on half the questions, reported on all; misses "Walmart employees"-type questions (needs LLM/entailment).
7. *"Last quarter" = 2018-Q2* because the dataset ends Aug 2018. Dates are data-relative, not today-relative.

**Known weaknesses (volunteer them)**: 37 eval questions; extractive answers; Portuguese theme rules are keywords;
vocabulary-gap failure ("people" vs "employees") documented as an xfail test; revenue definition excludes freight.

**If asked "what next?"** neural embeddings + cross-encoder comparison, LLM answer faithfulness eval (RAGAS),
larger labelled set, Postgres, per-user auth.
