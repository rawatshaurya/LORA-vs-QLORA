import os
import torch
import inspect
import json
import time

from trl import SFTTrainer

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig

from .utils import load_yaml, set_seed, ensure_dir
from .data import load_reasoning_dataset
from .metrics import extract_final_answer, exact_match


def build_model_and_tokenizer(cfg: dict):
    model_name = cfg["model_name"]
    use_4bit = bool(cfg.get("use_4bit", False))

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
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

    # NOTE: torch_dtype deprecation warning is harmless; transformers uses `dtype` in newer versions.
    torch_dtype = getattr(torch, cfg.get("torch_dtype", "float16"))

    # Enforce dtype at load-time (important for QLoRA on Windows)
    load_kwargs = dict(
        device_map="auto",
        quantization_config=quant_config,
    )

    try:
        # Newer Transformers prefers `dtype=`
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch_dtype,
            **load_kwargs,
        )
    except TypeError:
        # Older Transformers uses `torch_dtype=`
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            **load_kwargs,
        )

    model.config.use_cache = False
    return model, tokenizer



@torch.no_grad()
def quick_eval(model, tokenizer, eval_ds, cfg: dict, limit: int = 200):
    model.eval()
    n = min(limit, len(eval_ds))
    correct = 0

    gen_kwargs = dict(
        max_new_tokens=int(cfg.get("gen_max_new_tokens", 64)),
        temperature=float(cfg.get("gen_temperature", 0.0)),
        top_p=float(cfg.get("gen_top_p", 1.0)),
        do_sample=float(cfg.get("gen_temperature", 0.0)) > 0.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    for i in range(n):
        messages = eval_ds[i]["messages"]
        gold = eval_ds[i]["final_answer"]

        prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=int(cfg.get("max_seq_len", 1024)),
        ).to(model.device)

        out = model.generate(**inputs, **gen_kwargs)
        text = tokenizer.decode(out[0], skip_special_tokens=True)

        pred = extract_final_answer(text)
        correct += exact_match(pred, gold)

    return {"eval_exact_match": correct / n, "eval_n": n}


def main():
    cfg = load_yaml("configs/train.yaml")
    set_seed(int(cfg.get("seed", 42)))

    output_dir = cfg["output_dir"]
    ensure_dir(output_dir)

    # Load data
    train_ds = load_reasoning_dataset(
        cfg["dataset_name"],
        cfg.get("dataset_config"),
        cfg["train_split"],
        cfg["dataset_type"],
        cfg.get("max_train_samples"),
        int(cfg.get("seed", 42)),
    )
    eval_ds = load_reasoning_dataset(
        cfg["dataset_name"],
        cfg.get("dataset_config"),
        cfg["eval_split"],
        cfg["dataset_type"],
        cfg.get("max_eval_samples"),
        int(cfg.get("seed", 42)),
    )

    model, tokenizer = build_model_and_tokenizer(cfg)
    use_4bit = bool(cfg.get("use_4bit", False))
    use_cuda = torch.cuda.is_available()

    # Mixed precision:
    # - LoRA (not 4-bit): fp16 is fine
    # - QLoRA (4-bit) on Windows: disable AMP scaler path to avoid bf16 unscale issues
    fp16_flag = (not use_4bit) and use_cuda and (cfg.get("torch_dtype") == "float16")
    bf16_flag = False

    # ---- Make a plain-text field so SFTTrainer doesn't need tokenizer kwargs ----
    def add_text(example):
        example["text"] = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        return example

    train_ds = train_ds.map(add_text)
    eval_ds = eval_ds.map(add_text)

    # PEFT config (LoRA)
    lora_cfg = LoraConfig(
        r=int(cfg["lora_r"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(cfg.get("target_modules", ["q_proj", "v_proj"])),
    )
    # TrainingArguments is stable across TRL versions
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=float(cfg["num_train_epochs"]),
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        learning_rate=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
        warmup_ratio=float(cfg["warmup_ratio"]),
        lr_scheduler_type=str(cfg["lr_scheduler_type"]),
        max_grad_norm=(0.0 if use_4bit else float(cfg["max_grad_norm"])),
        logging_steps=int(cfg["logging_steps"]),
        save_steps=int(cfg["save_steps"]),
        save_total_limit=int(cfg["save_total_limit"]),
        bf16=bf16_flag,
        fp16=fp16_flag,
        report_to=[],
        remove_unused_columns=False,
        # evaluation_strategy="no",  # eval after training (older TRL-safe)
    )

    def formatting_func(example):
        return tokenizer.apply_chat_template(example["messages"], tokenize=False)

    def make_sft_trainer(**kwargs):
        sig = inspect.signature(SFTTrainer.__init__)
        allowed = set(sig.parameters.keys())
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
        return SFTTrainer(**filtered)

    trainer = make_sft_trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=None,                 # keep None for now (stable)
        peft_config=lora_cfg,
        dataset_text_field="text",         # many TRL versions support this
        max_seq_length=int(cfg.get("max_seq_len", 1024)),
        packing=False,
        # For newer TRL versions (if supported), this will be used; otherwise filtered out:
        processing_class=tokenizer,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    trainer.train()

    elapsed_sec = time.time() - start_time

    peak_alloc_mb = None
    peak_reserved_mb = None
    if torch.cuda.is_available():
        peak_alloc_mb = torch.cuda.max_memory_allocated() / (1024**2)
        peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024**2)

    # Save adapter
    adapter_dir = os.path.join(output_dir, "adapter")
    ensure_dir(adapter_dir)
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    # Quick eval
    metrics = quick_eval(trainer.model, tokenizer, eval_ds, cfg, limit=200)
    summary = {
        "run_name": os.path.basename(output_dir.rstrip("/\\") ),
        "output_dir": output_dir,
        "dataset_type": cfg.get("dataset_type"),
        "dataset_name": cfg.get("dataset_name"),
        "model_name": cfg.get("model_name"),
        "use_4bit": bool(cfg.get("use_4bit", False)),
        "torch_dtype": cfg.get("torch_dtype"),
        "lora_r": cfg.get("lora_r"),
        "lora_alpha": cfg.get("lora_alpha"),
        "target_modules": cfg.get("target_modules"),
        "max_seq_len": cfg.get("max_seq_len"),
        "train_samples": cfg.get("max_train_samples"),
        "eval_samples_quick": metrics.get("eval_n"),
        "quick_eval_exact_match": metrics.get("eval_exact_match"),
        "peak_vram_alloc_mb": peak_alloc_mb,
        "peak_vram_reserved_mb": peak_reserved_mb,
        "train_time_sec": elapsed_sec,
        }

    with open(os.path.join(output_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(output_dir, "metrics.txt"), "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    print("Training complete.")
    print("Saved adapter to:", adapter_dir)
    print("Quick eval:", metrics)


if __name__ == "__main__":
    main()
