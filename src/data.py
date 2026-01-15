from datasets import load_dataset

from .prompts import (
    SYSTEM_PROMPT,
    CSQA_USER_TEMPLATE,
    STRATEGYQA_USER_TEMPLATE,
)


def extract_gsm8k_final(answer: str) -> str:
    if "####" in answer:
        return answer.split("####")[-1].strip()
    return answer.strip().split()[-1]


def make_gsm8k_example(question: str, answer: str) -> dict:
    final = extract_gsm8k_final(answer)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Solve the problem.\n\nProblem:\n{question}\n\nReturn ONLY: Final Answer: <number>\n"},
        {"role": "assistant", "content": f"Final Answer: {final}"},
    ]
    return {"messages": messages, "final_answer": final}


def make_csqa_example(question: str, choices: dict, answer_key: str) -> dict:
    # choices: {'label': ['A'..], 'text': ['..'..]}
    pairs = list(zip(choices["label"], choices["text"]))
    choices_str = "\n".join([f"{lab}. {txt}" for lab, txt in pairs])

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": CSQA_USER_TEMPLATE.format(question=question, choices=choices_str)},
        {"role": "assistant", "content": f"Final Answer: {answer_key.strip().upper()}"},
    ]
    return {"messages": messages, "final_answer": answer_key.strip().upper()}


def make_strategyqa_example(question: str, answer_bool: bool) -> dict:
    gold = "yes" if bool(answer_bool) else "no"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": STRATEGYQA_USER_TEMPLATE.format(question=question)},
        {"role": "assistant", "content": f"Final Answer: {gold}"},
    ]
    return {"messages": messages, "final_answer": gold}


def load_reasoning_dataset(
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    dataset_type: str,
    max_samples: int | None,
    seed: int,
):
    """
    dataset_type in: {"gsm8k", "csqa", "strategyqa"}
    """
    if dataset_config:
        ds = load_dataset(dataset_name, dataset_config, split=split)
    else:
        ds = load_dataset(dataset_name, split=split)

    if max_samples is not None and max_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(max_samples))

    if dataset_type == "gsm8k":
        def _map(ex):
            return make_gsm8k_example(ex["question"], ex["answer"])
        return ds.map(_map, remove_columns=ds.column_names)

    if dataset_type == "csqa":
        def _map(ex):
            return make_csqa_example(ex["question"], ex["choices"], ex["answerKey"])
        return ds.map(_map, remove_columns=ds.column_names)

    if dataset_type == "strategyqa":
        # tasksource/strategy-qa has: question (str), answer (bool), plus others :contentReference[oaicite:3]{index=3}
        def _map(ex):
            return make_strategyqa_example(ex["question"], ex["answer"])
        return ds.map(_map, remove_columns=ds.column_names)

    raise ValueError(f"Unknown dataset_type: {dataset_type}")
