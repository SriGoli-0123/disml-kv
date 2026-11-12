"""QA quality metrics: HotpotQA-style EM/F1 and an optional LLM judge."""

from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def clean_prediction(text: str) -> str:
    """First line, strip common prefixes such as 'Answer:'."""
    t = text.strip()
    t = re.sub(r"^<think>.*?</think>", "", t, flags=re.S).strip()
    t = t.split("\n")[0].strip()
    t = re.sub(r"^(answer|final answer)\s*[:\-]\s*", "", t, flags=re.I)
    return t.strip().strip('"').strip("*").strip()


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    # yes/no/noanswer must match exactly (HotpotQA convention)
    if (p and p[0] in ("yes", "no", "noanswer")) or (g and g[0] in ("yes", "no", "noanswer")):
        if normalize_answer(pred) != normalize_answer(gold):
            return 0.0
    common = Counter(p) & Counter(g)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p)
    recall = num_same / len(g)
    return 2 * precision * recall / (precision + recall)


def contains(pred: str, gold: str) -> float:
    return float(normalize_answer(gold) in normalize_answer(pred))


def score_one(pred: str, gold) -> dict:
    """EM / F1 / contains for one prediction; ``gold`` may be a string or a list
    of acceptable answers (MuSiQue aliases) — the max over aliases is taken."""
    golds = [gold] if isinstance(gold, str) else [g for g in gold if g]
    p = clean_prediction(pred)
    return {
        "em": max(exact_match(p, g) for g in golds),
        "f1": max(f1_score(p, g) for g in golds),
        "contains": max(contains(pred, g) for g in golds),
    }


def evaluate(pairs) -> dict:
    """``pairs``: iterable of (prediction_text, gold_answer_or_alias_list)."""
    n = em = f1 = ct = 0
    for pred, gold in pairs:
        sc = score_one(pred, gold)
        n += 1
        em += sc["em"]
        f1 += sc["f1"]
        ct += sc["contains"]
    if n == 0:
        return {"n": 0}
    return {"n": n, "em": em / n, "f1": f1 / n, "contains": ct / n}


JUDGE_PROMPT = (
    "You are grading a question-answering system with a fixed rubric.\n"
    "Question: {question}\nReference answer: {gold}\nSystem answer: {pred}\n\n"
    "Does the system answer convey the same fact as the reference answer? "
    "Paraphrases and extra words are fine; a different entity, number, date or "
    "yes/no value is wrong. Reply with exactly one word: YES or NO."
)


async def llm_judge(session, base_url: str, model: str, question: str, gold: str, pred: str,
                    api_key: str | None = None) -> bool | None:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": JUDGE_PROMPT.format(question=question, gold=gold, pred=clean_prediction(pred) or pred)}],
        "max_tokens": 4,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        import os

        key = api_key or os.environ.get("JUDGE_API_KEY")
        headers = {"Authorization": f"Bearer {key}"} if key else None
        url = base_url.rstrip("/")
        url = url + ("/chat/completions" if url.endswith("/v1") else "/v1/chat/completions")
        r = await session.post(url, json=payload, timeout=120, headers=headers)
        r.raise_for_status()
        out = r.json()["choices"][0]["message"]["content"].strip().upper()
        return out.startswith("YES")
    except Exception:
        return None
