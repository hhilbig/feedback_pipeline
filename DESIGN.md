## Design Overview

This pipeline runs a rigorous, multi-step review process similar to a high-quality academic workshop.

### Pipeline Flow

```
Paper Text
    │
    ├─→ Generation Stage (with Perspective Seeds)
    │    └─→ Specialized Agents with unique focus areas
    │        (Theorists × 3, Methodologists × 2,
    │         Rival Researchers × 2, Editor × 1)
    │
    ├─→ Grounding Check
    │    └─→ Flag proposals referencing tables/figures/sections
    │        not found in the paper (hallucination guardrail)
    │
    ├─→ Scoring Stage (Dual-Pass + Confidence Weighting)
    │    ├─→ Score with manuscript first
    │    ├─→ Score with proposal first
    │    ├─→ Average scores to remove bias
    │    └─→ Adjust composite ±10% by judge agreement
    │
    ├─→ Selection Stage
    │    ├─→ Filter by quality thresholds
    │    └─→ Semantic deduplication (embedding cosine similarity)
    │
    ├─→ Critique & Revision Stage
    │    ├─→ Discussant Agent critiques top proposals
    │    └─→ Original Agents rewrite proposals → Revised Output
    │
    ├─→ Re-scoring & Merging
    │    └─→ Merge best Revised & Un-revised items
    │
    ├─→ Cluster & Synthesize
    │    └─→ Group related proposals by embedding similarity
    │        → Synthesize multi-proposal clusters
    │
    └─→ Synthesis Stage
         └─→ Meta-Reviewer → Final Report
             ├─→ Executive Summary
             └─→ Technical Implementation Plan
```

### Why this approach?

**1. Diverse Ensemble Generation** Relying on a single AI response is often hit-or-miss. Instead, we deploy "blocks" of specialized agents (Theorists, Methodologists, Rivals), each with a unique perspective seed that steers them toward different analytical focuses. Research shows structured prompt variation outperforms temperature-based stochasticity for ensemble diversity (Schoenegger et al., 2024; Wang et al., 2022).

**2. Fair Scoring (Bias Calibration)** AI models often prefer text simply because it appears first in the prompt. We fix this "positional bias" by scoring every proposal twice: once with the manuscript first, and once with the proposal first. Averaging these scores gives a much fairer signal of quality (Wang et al., 2024; Shi et al., 2024).

**3. Quality through Iteration (The Critique Loop)** First drafts are rarely perfect. Top proposals enter a "Discussant" loop where they are critiqued for vagueness or missing steps. The agents then rewrite their proposals to address these critiques. This mimics human peer review and significantly improves reasoning quality (Madaan et al., 2023).

**4. Actionable Output** The final step is not just a summary. The Meta-Reviewer converts the raw feedback into a Technical Implementation Plan with specific diagnostic steps (e.g., "Run a placebo test on pre-2020 data"), bridging the gap between identifying a problem and solving it.

### Scaling & Selection Details

**The "Block of 8" System** To maintain a balanced perspective as you scale, agents are added in blocks of 8. Each block contains:

- 3 Theorists: Focus on contribution and logic.
- 2 Rivals: Focus on alternative explanations.
- 2 Methodologists: Focus on empirical design.
- 1 Editor: Focus on clarity and structure.

**Scoring & Thresholds** We don't just accept everything. Proposals are ranked by a composite score: `0.35 × Importance + 0.25 × Specificity + 0.20 × Actionability + 0.20 × Uniqueness`

The composite score is then adjusted ±10% based on judge agreement: high-agreement proposals get a boost, contested proposals get a slight penalty. This confidence-weighted scoring reduces the number of samples needed while maintaining accuracy (CISC, 2025).

Only "High-Quality" proposals (Composite ≥ 3.0) make it to the critique stage. This ensures the final meta-review focuses only on the strongest, most actionable insights.

**Grounding Check** After generation but before scoring, proposals are checked for hallucinated references. A regex-based guardrail extracts references to tables, figures, sections, and other specific entities, then verifies they actually appear in the paper text. Flagged proposals are annotated (not removed) so the scorer and meta-reviewer can treat them with appropriate skepticism.

**Semantic Deduplication** Multiple agents often flag similar issues, even across different roles. We remove near-duplicate proposals using embedding cosine similarity (text-embedding-3-small, threshold ~0.82), which catches paraphrased duplicates that word-overlap methods miss. Deduplication is cross-dimensional, so a methodologist and theorist identifying the same underlying issue will be deduplicated.

**Cluster-then-Synthesize** Before meta-review, semantically related proposals are clustered using embedding similarity (threshold ~0.65). Multi-proposal clusters are synthesized into consolidated findings via one LLM call per cluster, preserving the strongest elements from each. This intermediate aggregation step improves final synthesis quality (Li et al., 2025) and reduces cognitive load on the meta-reviewer.

**Reviewer Agreement Signal** The dual-pass scoring produces not just averaged scores, but also a measure of agreement between passes. This "reviewer agreement" score (0-1) is passed to the meta-reviewer, who prioritizes high-consensus issues and notes uncertainty for contested assessments.

### Reliability Features

**Retry with Backoff** All API calls retry up to 3 times with exponential backoff (1s, 2s, 4s) on rate limits, timeouts, and connection errors.

**Partial Failure Recovery** If some proposal generations fail, the pipeline continues with successful results rather than crashing entirely.

### References

- **Gou, Z., Shao, Z., Gong, Y., et al. (2023).** CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing. *arXiv:2305.11738*.

- **Hossain, E., Sinha, S. K., Bansal, N., et al. (2025).** LLMs as Meta-Reviewers' Assistants: A Case Study. *NAACL 2025*.

- **Li, Z., et al. (2025).** Generative Self-Aggregation for LLM Ensembles.

- **Madaan, A., Tandon, N., Gupta, P., et al. (2023).** Self-Refine: Iterative Refinement with Self-Feedback. *arXiv:2303.17651*.

- **Schoenegger, P., et al. (2024).** Wisdom of the Silicon Crowd: LLM Ensemble Prediction Capabilities Rival Human Crowd Accuracy. *Science Advances*.

- **Shi, L., Ma, C., Liang, W., Ma, W., Vosoughi, S. (2024).** Judging the Judges: A Systematic Investigation of Position Bias in Pairwise Comparative Assessments by LLMs. *arXiv:2406.07791*.

- **Wang, P., Li, L., Chen, L., et al. (2024).** Large Language Models are not Fair Evaluators. *ACL 2024*.

- **Wang, X., Wei, J., Schuurmans, D., et al. (2022).** Self-Consistency Improves Chain of Thought Reasoning in Language Models. *arXiv:2203.11171*.

**Repo:** [https://github.com/hhilbig/feedback_pipeline](https://github.com/hhilbig/feedback_pipeline)
