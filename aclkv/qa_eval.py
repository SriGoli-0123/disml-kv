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


def evaluate(pairs) -> dict:
    """``pairs``: iterable of (prediction_text, gold_answer)."""
    n = em = f1 = ct = 0
    for pred, gold in pairs:
        p = clean_prediction(pred)
        n += 1
        em += exact_match(p, gold)
        f1 += f1_score(p, gold)
        ct += contains(pred, gold)
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


async def llm_judge(session, base_url: str, model: str, question: str, gold: str, pred: str) -> bool | None:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": JUDGE_PROMPT.format(question=question, gold=gold, pred=clean_prediction(pred) or pred)}],
        "max_tokens": 4,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        r = await session.post(f"{base_url.rstrip('/')}/v1/chat/completions", json=payload, timeout=120)
        r.raise_for_status()
        out = r.json()["choices"][0]["message"]["content"].strip().upper()
        return out.startswith("YES")
    except Exception:
        return None
