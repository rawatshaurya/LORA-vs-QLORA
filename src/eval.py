import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from .utils import load_yaml, set_seed, ensure_dir
from .data import load_reasoning_dataset
from .metrics import extract_final_answer, exact_match


def load_base_and_adapter(cfg: dict):
    model_name = cfg["model_name"]
    adapter_dir = cfg["adapter_dir"]
    use_4bit = bool(cfg.get("use_4bit", False))

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if use_4bit:
        compute_dtype = getattr(torch, cfg.get("bnb_4bit_compute_dtype", "float16"))
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quant_config,
    )
    base.config.use_cache = True

    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def evaluate(model, tokenizer, eval_ds, cfg: dict):
    gen_kwargs = dict(
        max_new_tokens=int(cfg.get("gen_max_new_tokens", 256)),
        temperature=float(cfg.get("gen_temperature", 0.0)),
        top_p=float(cfg.get("gen_top_p", 1.0)),
        do_sample=float(cfg.get("gen_temperature", 0.0)) > 0.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    correct = 0
    rows = []

    for i in range(len(eval_ds)):
        messages = eval_ds[i]["messages"]
        gold = eval_ds[i]["final_answer"]

        prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=int(cfg["max_seq_len"])).to(model.device)

        out = model.generate(**inputs, **gen_kwargs)
        text = tokenizer.decode(out[0], skip_special_tokens=True)

        pred = extract_final_answer(text)
        em = exact_match(pred, gold)
        correct += em

        rows.append({
            "idx": i,
            "gold": gold,
            "pred": pred,
            "exact_match": em,
            "output_text": text[-1200:],  # keep last part to avoid massive files
        })

    return correct / len(eval_ds), rows


def main():
    cfg = load_yaml("configs/eval.yaml")
    set_seed(int(cfg.get("seed", 42)))

    out_path = cfg["output_path"]
    ensure_dir("/".join(out_path.split("/")[:-1]) or ".")

    eval_ds = load_reasoning_dataset(
        cfg["dataset_name"],
        cfg.get("dataset_config"),
        cfg["eval_split"],
        cfg["dataset_type"],
        cfg.get("max_eval_samples"),
        int(cfg.get("seed", 42)),
    )


    model, tokenizer = load_base_and_adapter(cfg)
    acc, rows = evaluate(model, tokenizer, eval_ds, cfg)

    payload = {"exact_match": acc, "n": len(eval_ds), "rows": rows}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Exact Match: {acc:.4f} on n={len(eval_ds)}")
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
