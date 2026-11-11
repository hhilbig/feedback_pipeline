# feedback_llm Status

## Current State

The private-feedback benchmark infrastructure is implemented. It imports
hash-bound manuscript-review pairs, prepares private adjudication packets, runs
a feedback-free cold baseline behind an explicit cost ceiling, and emits only
pseudonymous aggregate results outside the private corpus directory. Reviewer
prompts, model routing, reasoning settings, thresholds, and report synthesis
have not changed in this milestone.

The five-family source freeze is complete. All manuscript and feedback files
are materialized, copied into the private corpus, and hash-verified. The offline
import produces 12 human-review records, 150 issue candidates, and 145 clusters.
The gold-adjudication packet is partly complete. A human has completed 52 of 145
cluster rows. All 36 rows that currently require full adjudication are complete;
30 are included scoring targets, 16 are completed exclusions, and six are
tier-screened minor rows outside the deterministic sample. The remaining 93
rows still need a tier screen before the strict benchmark can run.

The cold-baseline dry run estimated a total API cost of $9.7836. The first paid
attempt cost $9.8542 and kept historical feedback out of every prompt. It returned
5, 5, 4, 4, and 5 final issues across the five cases. The old exact-five audit gate
therefore marked the run incomplete even though all manuscripts were evaluated.
That attempt retained cost and run metadata but not enough untruncated issue text
to create a defensible adjudication packet.

The benchmark now accepts one to five real final issues per case. Missing slots
count as misses for supported-significant precision@5 and valid-novelty yield@5;
duplicate rate remains conditional on returned issues. The runner writes a private
full-text checkpoint after each completed case and before the final audit.

The rerun on commit `f2cfe1d` completed all five cases and cost $9.9614, below
the approved $10 ceiling. It returned five issues per case, creating a 25-row
canonical generated-issue packet. Every split records zero review-memory records
and zero review-memory issues. The canonical packet and content-addressed checkpoint
have matching bindings, all private artifacts have the required permissions, and
the portable result contains no private paths or issue text. The run is now
`pending_human_adjudication`.

One journal source includes only verbatim reviewer passages. Its author responses
and resubmission remain hash-bound deferred sources and are not evaluation targets.

## Provider and Routing Change (2026-11-11)

API calls now go through OpenRouter by default (`LLM_PROVIDER=openai` restores direct
OpenAI). The default routing moved from GPT-5.6 to the GPT-6.1 family with reasoning on
for generation, verification, triage, and the report. The completed baseline's routing
is preserved as the `baseline-gpt-5.6` preset. Benchmark commands that must match the
completed baseline need `--routing baseline-gpt-5.6`; without it they run the new
routing. The default also rotates the eight reviewers across three vendors' models and
lets each return up to two critiques (the baseline preset does neither). None of this
has been evaluated on the private benchmark, and no full paper has been run through it. Details are in DESIGN.md under Models and Routing.

## Implemented

1. Manifest v1 with family-level leakage boundaries, ordered manuscript
   bundles, source hashes, pseudonymous reviewers, provenance, selectors,
   benchmark tiers, and source dispositions.
2. Offline extraction for PDF annotations, Word comments, review PDFs, and
   Markdown, text, or Word body feedback. The importer rejects stale hashes,
   unsupported files, missing files, and cloud placeholders.
3. Deterministic issue atomization, source deduplication, provisional
   clustering, major-cluster screening, and a five-minor-cluster sample per
   family using seed `20260802`.
4. Hash-bound private gold and generated-issue adjudication packets. Changes to
   source content, extraction rules, cluster membership, output-cardinality
   policy, or baseline output invalidate the corresponding labels. Incremental
   and content-addressed checkpoints preserve full generated text and evidence
   IDs without entering portable output.
5. Cold evaluation mode that passes neither a review corpus nor a reviewer
   prior to the pipeline and scores only the final post-verification issues, up
   to five per case.
6. Zero-call dry runs, finite positive cost ceilings, complete preflight cost
   checks, fixed pilot-composition checks before the first request, and
   remaining-budget checks before later cases.
7. Portable output projection that removes paths, reviewer identities, source
   text, generated text, locators, and notes.
8. Tests for extractors, mixed human/generated feedback, deduplication,
   family holdout, cold-run isolation, deterministic sampling, tamper detection,
   cost gates, partial-run rejection, and output privacy.
9. An explicit partial-gold pilot mode that freezes completed labels, scores only
   the 30 fully adjudicated included clusters, and labels major recall as
   non-exhaustive. The strict complete-gold gate remains the default.
10. An up-to-five output policy with fixed five-slot denominators for supported
    precision and novelty yield, unchanged human-cluster recall denominators,
    and returned-issue denominators for duplicate rates.
11. A local generated-issue labeling screen with binding verification, guarded
    draft saves, manuscript-rank navigation, same-family human matches, and
    same-case duplicate choices. The existing screen remains available for gold
    feedback adjudication.

## Source Freeze

| Family | Current status | Pilot treatment |
|---|---|---|
| Family 1 | Ready | Primary |
| Family 2 | Ready | Primary, journal |
| Family 3 | Ready | Primary, journal |
| Family 4 | Ready | Primary, journal |
| Family 5 | Ready | Secondary only |

Family identities and source details are kept in the private corpus, not in this
repository.

## Next Validation Gate

1. Label all 25 rows in the generated-issue packet for correctness, significance,
   evidence sufficiency, human-cluster match, duplication, and valid novelty.
   The local adjudication app opens this queue by default, one page per paper.
   Each critique gets two answers: a verdict and a reviewer match (pre-filled
   with the lexical proposal). The verdict sets three fields at once: valid and
   important = correct/significant/sufficient; valid but minor =
   correct/minor/sufficient; wrong or not supported =
   incorrect/not_significant/insufficient; can't tell = unclear. Wrong and
   unsupported critiques are therefore not distinguished. Valid novelty is
   derived (valid, unmatched, unique), not asked.
2. Finalize explicitly partial,
   non-exhaustive major-recall results plus sampled-minor, precision, novelty,
   duplicate, and cost metrics.
3. Use the adjudicated baseline to decide whether prompt, routing, or threshold
   changes are justified. Do not tune the pipeline against unadjudicated output.
4. Screen the remaining 93 rows later if an exhaustive benchmark is still worth
   the additional labeling time.

Do not change reviewer prompts, models, routing, reasoning effort, thresholds,
or synthesis until the completed baseline has been human-adjudicated.
