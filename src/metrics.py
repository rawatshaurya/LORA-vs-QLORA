import re

# Supports:
#   Final Answer: A
#   Final Answer: yes
#   Final Answer: 42  (still works if you ever go back to GSM8K)
_FINAL_RE = re.compile(r"Final Answer:\s*([A-Ea-e]|yes|no|[-+]?\d+(\.\d+)?)", re.IGNORECASE)


def extract_final_answer(text: str) -> str | None:
    m = _FINAL_RE.search(text)
    if not m:
        return None
    return m.group(1).strip()


def normalize_answer(ans: str | None) -> str | None:
    if ans is None:
        return None
    a = ans.strip()
    # normalize letters
    if len(a) == 1 and a.lower() in ["a", "b", "c", "d", "e"]:
        return a.upper()
    # normalize yes/no
    if a.lower() in ["yes", "no"]:
        return a.lower()
    # numbers
    return a


def exact_match(pred: str | None, gold: str) -> int:
    p = normalize_answer(pred)
    g = normalize_answer(gold)
    if p is None or g is None:
        return 0
    return int(p == g)
