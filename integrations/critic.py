"""Critic agent: reviews each round with a non-Anthropic model via OpenRouter.

Using a different model family from the Hypothesis Generator makes the two
models' mistakes less correlated.
"""
import json
import os
import re

from dotenv import load_dotenv

from integrations.llm import openrouter_chat

load_dotenv()

SYSTEM = """You are a skeptical materials scientist reviewing one round of an autonomous lab searching for lithium superionic solid electrolytes.

You receive the round record: active hypotheses, the experiment design the planner chose (and the alternatives it scored), the measurements (log10 conductivity in S/cm), and the running totals.

Check:
1. Do the conclusions follow from the measurements? The OBELiX paper reports ~0.41 experimental uncertainty in log sigma and repeat measurements of the same formula scatter by ~0.66; treat differences below ~0.7 as noise.
2. Is any claim overreaching (few measurements, one family, near-duplicate materials counted as separate discoveries)?
3. Was a hypothesis contradicted by the data? If so, it must be reopened.
4. Is the planner exploiting too early or exploring too long given the remaining budget?

Return only JSON:
{"verdict": "accept" | "reopen",
 "hypothesis_ids": [ids to reopen],
 "reasons": [short strings],
 "suggestion": "one sentence for the planner's next round",
 "confidence": number between 0 and 1}"""


def critic(round_record, model=None):
    model = model or os.getenv("CRITIC_MODEL", "google/gemini-3.8-flash")
    # The Hypothesis Generator runs on Mistral, so the Critic must not.
    if "mistral" in model.lower():
        raise ValueError("The Critic must use a different model family from the Mistral Hypothesis Generator")
    data = openrouter_chat({
        "model": model,
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(round_record, default=str)},
        ],
    })
    content = data["choices"][0]["message"]["content"]
    match = re.search(r"\{.*\}", content, re.S)
    review = json.loads(match.group(0) if match else content)
    review.setdefault("verdict", "accept")
    review.setdefault("hypothesis_ids", [])
    review["model"] = model
    return review


def review_round(run_id, round_, round_record):
    """Run the Critic, log it, and mark reopened hypotheses in Supabase."""
    from integrations.supabase_sync import db

    review = critic(round_record)
    kind = "reopen" if review["verdict"] == "reopen" else "review"
    summary = "; ".join(review.get("reasons", [])[:2]) or review["verdict"]
    db.event(run_id, round_, "critic", kind, summary, review)
    for hid in review["hypothesis_ids"]:
        db.update("hypotheses", {"id": hid}, {"status": "reopened"})
    return review


if __name__ == "__main__":
    demo = {
        "round": 3,
        "hypotheses": [{"id": "h1", "statement": "Garnet-type oxides dominate the high-conductivity region", "confidence": 0.7}],
        "chosen_design": "exploit",
        "measurements": [{"material": "Li6.5La3Zr1.5Ta0.5O12", "family": "garnet", "log_sigma": -3.6},
                         {"material": "Li7La3Zr2O12", "family": "garnet", "log_sigma": -4.1}],
        "found_so_far": 0, "measured": 15, "budget": 50,
    }
    print(json.dumps(critic(demo), indent=2))
