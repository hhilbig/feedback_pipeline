## Design Overview

This pipeline implements a **reviewer-ensemble architecture** designed to mimic a rigorous academic review process. Instead of relying on a single stochastic generation, we deploy scalable "blocks" of specialized agents (Theorists, Methodologists, Rival Researchers, Editors). This ensures diverse coverage of the manuscript's weaknesses—balancing conceptual logic with empirical rigor—and reduces the idiosyncratic failures common in single-shot LLM interactions (Wang et al., 2022).

To address the instability of LLM-based evaluation, the scoring stage is **calibrated against positional bias**. Research shows LLMs often favor content presented in specific orders (e.g., first vs. last). We mitigate this via a **dual-pass scoring mechanism**: every proposal is evaluated twice—once with the manuscript preceding the feedback, and once with the order reversed. The averaged score provides a more robust signal of quality (Wang et al., 2024; Shi et al., 2024).

Quality is further enhanced through an **iterative refinement loop**. High-scoring proposals are not accepted as-is; they are challenged by a separate "Discussant" agent that identifies vague claims or missing steps. The original agents then rewrite their proposals to address these critiques. This aligns with frameworks showing that explicit critique and revision significantly improve reasoning performance (Madaan et al., 2023; Gou et al., 2023). Crucially, the pipeline merges these revised outputs with the best un-revised proposals to avoid "survivor bias," ensuring the final selection represents the strongest possible set of insights.

Finally, the synthesis layer is architected to prioritize **actionability over summary**. Rather than a passive overview, the Meta-Reviewer is prompted to generate a structured **Technical Implementation Plan** with diagnostic next steps. This bridges the gap between identifying a problem (e.g., "endogeneity concern") and solving it (e.g., "run a placebo test using pre-treatment data").

### References

* **Gou, Z., Shao, Z., Gong, Y., et al. (2023).** CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing. *arXiv:2305.11738*.

* **Hossain, E., Sinha, S. K., Bansal, N., et al. (2025).** LLMs as Meta-Reviewers' Assistants: A Case Study. *NAACL 2025*.

* **Madaan, A., Tandon, N., Gupta, P., et al. (2023).** Self-Refine: Iterative Refinement with Self-Feedback. *arXiv:2303.17651*.

* **Shi, L., Ma, C., Liang, W., Ma, W., Vosoughi, S. (2024).** Judging the Judges: A Systematic Investigation of Position Bias in Pairwise Comparative Assessments by LLMs. *arXiv:2406.07791*.

* **Wang, P., Li, L., Chen, L., et al. (2024).** Large Language Models are not Fair Evaluators. *ACL 2024*.

* **Wang, X., Wei, J., Schuurmans, D., et al. (2022).** Self-Consistency Improves Chain of Thought Reasoning in Language Models. *arXiv:2203.11171*.

**Repo:** [https://github.com/hhilbig/feedback_llm](https://github.com/hhilbig/feedback_llm)

