import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from .prompts import SYSTEM_PROMPT, USER_TEMPLATE
from .metrics import extract_final_answer


def load_model(model_name: str, adapter_dir: str, use_4bit: bool, compute_dtype: str):
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if use_4bit:
        cdtype = getattr(torch, compute_dtype)
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=cdtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quant_config,
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def run_one(model, tokenizer, question: str, max_seq_len: int, max_new_tokens: int, temperature: float, top_p: float):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(question=question)},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_len).to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=temperature > 0.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    final = extract_final_answer(text)
    return text, final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter_dir", type=str, default="outputs/run_1/adapter")
    ap.add_argument("--use_4bit", action="store_true")
    ap.add_argument("--compute_dtype", type=str, default="bfloat16")
    ap.add_argument("--max_seq_len", type=int, default=1024)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--question", type=str, required=True)
    args = ap.parse_args()

    model, tokenizer = load_model(args.model_name, args.adapter_dir, args.use_4bit, args.compute_dtype)
    text, final = run_one(
        model, tokenizer, args.question,
        args.max_seq_len, args.max_new_tokens,
        args.temperature, args.top_p
    )

    print(text)
    print("\n----")
    print("Parsed Final Answer:", final)


if __name__ == "__main__":
    main()
