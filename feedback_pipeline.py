import asyncio
import json
import sys
from argparse import ArgumentParser
from typing import List, Dict, Any

from openai import AsyncOpenAI


"""
Very lightweight async feedback pipeline.

High-level steps:
1. Generate proposals (8 independent workers).
2. Score each proposal (4 criteria).
3. Rank and classify proposals in Python only.
4. Produce a meta-review from all high-quality proposals.

Details of prompts / thresholds are intentionally minimal for now
and can be refined later.
"""


client = AsyncOpenAI()  # Requires OPENAI_API_KEY in environment


# Thresholds for meta-review inclusion (tune as needed)
IMPORTANCE_THRESHOLD = 3
COMPOSITE_THRESHOLD = 3.0
TOP_K = 3

DIMENSIONS = [
    "contribution",
    "logical_soundness",
    "interpretation",
    "writing_structure",
]


# -------------------------------------------------------------------
# Helper: generic JSON chat call
# -------------------------------------------------------------------


async def chat_json(
    messages: List[Dict[str, str]],
    model: str = "gpt-5.1-mini",
) -> Any:
    """Call the chat API and parse a JSON object response."""
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content
    return json.loads(content)


# -------------------------------------------------------------------
# 1. Independent generation workers
# -------------------------------------------------------------------


async def generate_single_proposal(paper_text: str, worker_id: int) -> Dict[str, Any]:
    system_msg = {
        "role": "system",
        "content": (
            "You are an expert reviewer for quantitative social science papers. "
            "You produce a single, concise, high-impact feedback proposal."
        ),
    }
    user_msg = {
        "role": "user",
        "content": f"""
You review the paper text below and provide exactly one feedback proposal.

Your goal is to identify the single most important change that would improve the paper.
Choose one of the following dimensions that best fits your feedback:
- "contribution": novelty, substantive importance, positioning in the literature.
- "logical_soundness": logical coherence and internal consistency of the argument.
- "interpretation": interpretation of empirical results and their connection to theory.
- "writing_structure": clarity of exposition, organization, and structure.

Requirements for the feedback proposal:
- Maximum 3 sentences.
- Focus on the single most important issue you see.
- Reference at least one concrete element of the text (for example, a section, claim, figure, or type of analysis).
- Use neutral, precise, and technical language.
- Make the proposal directly actionable if possible.

Return a JSON object with fields:
- "id": integer worker id ({worker_id})
- "dimension": one of ["contribution","logical_soundness","interpretation","writing_structure"]
- "text": the feedback text

Paper text:
```text
{paper_text}
```""",
    }

    result = await chat_json([system_msg, user_msg])
    result["id"] = worker_id  # enforce id
    return result


async def generate_all_proposals(
    paper_text: str,
    n_workers: int = 8,
) -> List[Dict[str, Any]]:
    tasks = [
        generate_single_proposal(paper_text, worker_id=i)
        for i in range(1, n_workers + 1)
    ]
    proposals = await asyncio.gather(*tasks)
    return proposals


# -------------------------------------------------------------------
# 2. Independent scoring workers
# -------------------------------------------------------------------


async def score_single_proposal(
    paper_text: str,
    proposal: Dict[str, Any],
) -> Dict[str, Any]:
    system_msg = {
        "role": "system",
        "content": (
            "You evaluate the quality of a single feedback proposal for a "
            "social science paper. You assign integer scores only."
        ),
    }
    user_msg = {
        "role": "user",
        "content": f"""
You receive the paper text and one feedback proposal.

Assign four integer scores from 1 to 5:
- "importance": impact of the feedback on improving the paper.
- "specificity": degree of grounding in concrete, identifiable parts of the text.
- "actionability": clarity of what the author should change based on this feedback.
- "uniqueness": distinctiveness relative to typical comments on such papers.

Return a JSON object:
- "importance": integer 1–5
- "specificity": integer 1–5
- "actionability": integer 1–5
- "uniqueness": integer 1–5

Paper text:
```text
{paper_text}

Feedback proposal (dimension = {proposal.get("dimension")}):

{proposal.get("text")}
```""",
    }

    scores = await chat_json([system_msg, user_msg])
    scored = {
        **proposal,
        "importance": int(scores["importance"]),
        "specificity": int(scores["specificity"]),
        "actionability": int(scores["actionability"]),
        "uniqueness": int(scores["uniqueness"]),
    }
    return scored


async def score_all_proposals(
    paper_text: str,
    proposals: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    tasks = [score_single_proposal(paper_text, p) for p in proposals]
    scored = await asyncio.gather(*tasks)

    # Compute composite deterministically in Python
    for s in scored:
        I = s["importance"]
        S = s["specificity"]
        A = s["actionability"]
        U = s["uniqueness"]
        composite = 0.4 * I + 0.3 * S + 0.2 * A + 0.1 * U
        s["composite"] = composite

    return scored


# -------------------------------------------------------------------
# 3. Deterministic selection, ranking, thresholds
# -------------------------------------------------------------------


def select_and_classify(scored: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Sort by composite, descending
    sorted_scored = sorted(scored, key=lambda x: x["composite"], reverse=True)

    # Top K
    top_proposals = sorted_scored[:TOP_K]

    # Low-value proposals
    low_value_ids = [
        p["id"]
        for p in sorted_scored
        if p["importance"] <= 2 or p["actionability"] <= 2
    ]

    # High-quality proposals for meta-review
    high_quality = [
        p
        for p in sorted_scored
        if (p["composite"] >= COMPOSITE_THRESHOLD)
        or (p["importance"] >= IMPORTANCE_THRESHOLD)
    ]

    # Group high-quality proposals by dimension
    by_dimension = {dim: [] for dim in DIMENSIONS}
    for p in high_quality:
        dim = p.get("dimension")
        if dim in by_dimension:
            by_dimension[dim].append(p)

    selection = {
        "sorted": sorted_scored,
        "top_proposals": top_proposals,
        "low_value_ids": low_value_ids,
        "high_quality": high_quality,
        "by_dimension": by_dimension,
    }
    return selection


# -------------------------------------------------------------------
# 4. Meta-review using all high-quality proposals
# -------------------------------------------------------------------


async def meta_review(selection: Dict[str, Any]) -> str:
    by_dim_payload = {
        dim: [
            {
                "id": p["id"],
                "text": p["text"],
                "importance": p["importance"],
                "specificity": p["specificity"],
                "actionability": p["actionability"],
                "uniqueness": p["uniqueness"],
                "composite": p["composite"],
            }
            for p in plist
        ]
        for dim, plist in selection["by_dimension"].items()
    }

    top_global = selection["sorted"][:TOP_K]
    top_global_payload = [
        {
            "id": p["id"],
            "dimension": p["dimension"],
            "text": p["text"],
            "composite": p["composite"],
        }
        for p in top_global
    ]

    system_msg = {
        "role": "system",
        "content": (
            "You are writing a concise meta-review that synthesizes selected "
            "feedback proposals on a quantitative social science paper."
        ),
    }
    user_msg = {
        "role": "user",
        "content": f"""
You receive high-quality feedback proposals grouped by dimension, and the globally strongest proposals.

Dimensions:
- contribution
- logical_soundness
- interpretation
- writing_structure

Write a meta-review with four sections, in this order:
1. Contribution
2. Logical soundness of the argument
3. Interpretation of empirical results
4. Writing and structure

For each section:
- If there are proposals for that dimension, write 2–3 sentences that integrate their content and provide directly actionable guidance.
- If there are no proposals for that dimension, write 1–2 sentences indicating that no major issues were flagged there.

Then provide a numbered list of the three most important revisions across all dimensions, ordered from most to least important. Base this list primarily on the globally strongest proposals, but you may merge or rephrase them for clarity.

High-quality proposals by dimension:
```json
{json.dumps(by_dim_payload)}

Globally strongest proposals:
{json.dumps(top_global_payload)}
```""",
    }

    resp = await client.chat.completions.create(
        model="gpt-5.1",
        messages=[system_msg, user_msg],
    )
    return resp.choices[0].message.content


# -------------------------------------------------------------------
# 5. Full pipeline wrapper + convenience entry point
# -------------------------------------------------------------------


async def full_feedback_pipeline(paper_text: str) -> Dict[str, Any]:
    """Run the full async feedback pipeline for a single paper."""
    proposals = await generate_all_proposals(paper_text)
    scored = await score_all_proposals(paper_text, proposals)
    selection = select_and_classify(scored)
    meta = await meta_review(selection)

    return {
        "proposals": proposals,
        "scored": scored,
        "selection": selection,
        "meta_review": meta,
    }


def feedback(paper_text: str) -> str:
    """
    Synchronous convenience wrapper.

    Returns only the meta-review text. For more detailed inspection
    (scores, selection, etc.), use `full_feedback_pipeline` directly.
    """
    return asyncio.run(full_feedback_pipeline(paper_text))["meta_review"]


__all__ = [
    "full_feedback_pipeline",
    "feedback",
    "generate_all_proposals",
    "score_all_proposals",
    "select_and_classify",
    "meta_review",
]
@@
    "meta_review",
]


def _read_paper_from_stdin() -> str:
    return sys.stdin.read()


def _read_paper_from_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main(argv: List[str] | None = None) -> int:
    """
    Minimal CLI entry point.

    Usage examples:
      python -m feedback_pipeline --file paper.txt
      cat paper.txt | python -m feedback_pipeline
    """
    parser = ArgumentParser(description="Run the feedback pipeline on a paper.")
    parser.add_argument(
        "--file",
        "-f",
        help="Path to a text file containing the paper. If omitted, read from stdin.",
    )
    args = parser.parse_args(argv)

    if args.file:
        paper_text = _read_paper_from_file(args.file)
    else:
        paper_text = _read_paper_from_stdin()

    if not paper_text.strip():
        print("No paper text provided (file was empty or stdin had no content).", file=sys.stderr)
        return 1

    meta_review = feedback(paper_text)
    print(meta_review)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
