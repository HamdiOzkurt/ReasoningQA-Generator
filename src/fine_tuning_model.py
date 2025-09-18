
import os
import torch
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import json
import re
import gc
from unsloth import FastLanguageModel

# --- KONFİGÜRASYON ---
CONFIG = {
    "model_name": "unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit",
    "dataset_path": "data/test_qa.json",
    "output_dir": "dora-finetuned-r1-oracle",
    "max_seq_length": 2048,
    "dora_rank": 16,
    "dora_alpha": 32,
    "batch_size": 1,
    "gradient_accumulation_steps": 8,
    "epochs": 2,
    "learning_rate": 1e-4
}

# --- VERİ YÜKLEME ---
def load_data(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [{"input": x["question"], "output": x["answer"]} for x in data]
    except Exception as e:
        print(f"Veri yükleme hatası: {e}")
        return []

# --- MODEL EĞİTİMİ ---
def train_model():
    # Veri yükle
    print("Veri yükleniyor...")
    data = load_data(CONFIG["dataset_path"])
    dataset = Dataset.from_list(data)
    
    # Model yükle
    print("Model yükleniyor...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=CONFIG["model_name"],
        max_seq_length=CONFIG["max_seq_length"],
        load_in_4bit=True,
    )
    
    # DoRA adaptörü ekle
    model = FastLanguageModel.get_peft_model(
        model,
        r=CONFIG["dora_rank"],
        lora_alpha=CONFIG["dora_alpha"],
        lora_dropout=0.1,
        bias="none",
        use_gradient_checkpointing="unsloth",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_dora=True,
    )
    
    # Eğitim argümanları
    training_args = TrainingArguments(
        per_device_train_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        warmup_steps=10,
        num_train_epochs=CONFIG["epochs"],
        learning_rate=CONFIG["learning_rate"],
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        output_dir="outputs",
        save_steps=100,
        save_total_limit=3,
    )
    
    # Eğitici
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="input",
        max_seq_length=CONFIG["max_seq_length"],
        args=training_args,
    )
    
    # Eğitimi başlat
    print("Eğitim başlıyor...")
    trainer.train()
    
    # Modeli kaydet
    output_dir = CONFIG["output_dir"]
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model kaydedildi: {output_dir}")
    
    return model, tokenizer

if __name__ == "__main__":
    train_model()
