## Design Overview

This pipeline runs a rigorous, multi-step review process similar to a high-quality academic workshop.

### Pipeline Flow

```
Paper Text
    │
    ├─→ Generation Stage
    │    └─→ Specialized Agents (Theorists, Methodologists, 
    │        Rival Researchers, Editors) → Proposals
    │
    ├─→ Scoring Stage (Dual-Pass)
    │    ├─→ Score with manuscript first
    │    ├─→ Score with proposal first
    │    └─→ Average scores to remove bias
    │
    ├─→ Selection Stage
    │    └─→ Select Top-K Proposals
    │
    ├─→ Critique & Revision Stage
    │    ├─→ Discussant Agent critiques top proposals
    │    └─→ Original Agents rewrite proposals → Revised Output
    │
    ├─→ Re-scoring & Merging
    │    └─→ Merge best Revised & Un-revised items
    │
    └─→ Synthesis Stage
         └─→ Meta-Reviewer → Final Report
             ├─→ Executive Summary
             └─→ Technical Implementation Plan
```

### Why this approach?

**1. Strength in Numbers (Ensemble Generation)** Relying on a single AI response is often hit-or-miss. Instead, we deploy "blocks" of specialized agents (Theorists, Methodologists, Rivals). This ensures we catch different types of errors—from logical gaps to statistical flaws—and reduces the random hallucinations common in single-shot prompts (Wang et al., 2022).

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

Only "High-Quality" proposals (Composite ≥ 3.0) make it to the critique stage. This ensures the final meta-review focuses only on the strongest, most actionable insights.

### References

- **Gou, Z., Shao, Z., Gong, Y., et al. (2023).** CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing. *arXiv:2305.11738*.

- **Hossain, E., Sinha, S. K., Bansal, N., et al. (2025).** LLMs as Meta-Reviewers' Assistants: A Case Study. *NAACL 2025*.

- **Madaan, A., Tandon, N., Gupta, P., et al. (2023).** Self-Refine: Iterative Refinement with Self-Feedback. *arXiv:2303.17651*.

- **Shi, L., Ma, C., Liang, W., Ma, W., Vosoughi, S. (2024).** Judging the Judges: A Systematic Investigation of Position Bias in Pairwise Comparative Assessments by LLMs. *arXiv:2406.07791*.

- **Wang, P., Li, L., Chen, L., et al. (2024).** Large Language Models are not Fair Evaluators. *ACL 2024*.

- **Wang, X., Wei, J., Schuurmans, D., et al. (2022).** Self-Consistency Improves Chain of Thought Reasoning in Language Models. *arXiv:2203.11171*.

**Repo:** [https://github.com/hhilbig/feedback_pipeline](https://github.com/hhilbig/feedback_pipeline)
