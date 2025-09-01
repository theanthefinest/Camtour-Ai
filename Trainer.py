# === Libraries ===>

from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from huggingface_hub import login

# === Huggin_Face Token ===>
login(token="your_huggingface_token")

# === Data Preparation and Preprocessing ===>

# === Load Dataset ===
dataset = load_dataset('json', data_files="your_data_path")

# === Load Tokenizer ===
base_model = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(base_model)

# === Set PAD token if missing ===
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# === Apply Chat Format ===
def chat_format(examples):
    return {
        "prompt": tokenizer.apply_chat_template(examples["messages"], tokenize=False)
    }

formatted_dataset = dataset.map(chat_format)

# === Tokenize Dataset ===
def tokenize_function(examples):
    return tokenizer(
        examples["prompt"],
        truncation=True,
        padding=True,
        max_length=1024
    )

tokenized_dataset = formatted_dataset.map(tokenize_function, batched=True)

# === Data Collator ===
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# ====== End Data Preparation and Preprocessing ======


# === Bits and Bytes Configuration ===>

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# === Model Configuration ===>

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# === Merge existing LoRA ===>

resume_model_path = "your_model_path_checkpoint" # if you want to continue your training from the past trained model 
temp_model = PeftModel.from_pretrained(model, resume_model_path)
model = temp_model.merge_and_unload()

# === Apply new LoRA configuration ===> # if you don't want to apply new LoRA configuration, comment this block
new_lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# === Prepare for stabilities ===>
from peft import prepare_model_for_kbit_training
model = prepare_model_for_kbit_training(model)

# === Apply LoRA to the model ===>

# if no new LoRA configuration is needed, replace new_lora_config with resume_model_path
model = get_peft_model(model, new_lora_config) 
model.train()
model.config.use_cache = False

# === Check the Trainable Parameters ===

print(f"Model training mode: {model.training}")
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params}")
print(f"Active adapters: {model.active_adapter}")

# === Training Arguments Configuration ===
training_args = TrainingArguments(
    output_dir="/content/drive/MyDrive/Colab Notebooks/Model/chatbot_v3", # Your save path 
    per_device_train_batch_size=2,               # GPU memory constraint
    gradient_accumulation_steps=16,              #from transformers import TrainingArguments Effective batch size of 16
    learning_rate=1.5e-4,                        # Slightly reduced LR for stability
    num_train_epochs=3,
    fp16=True,                                   # T4 supports fp16
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    optim="paged_adamw_8bit",                    # Better performance on T4
    warmup_ratio=0.1,
    report_to=None,
    gradient_checkpointing=True,
    max_grad_norm=0.3,
    lr_scheduler_type="cosine",
    logging_first_step=True,
    seed=42,
    dataloader_pin_memory=True,                  # Re-enable pinning for performance
    remove_unused_columns=True,
    save_safetensors=True,
)

# === Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
)


print("=== Final verification ===")
print(f"Model training mode: {model.training}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print(f"Dataset size: {len(tokenized_dataset['train'])}")


try:
    print("Starting training...")
    trainer.train()
    print("Training completed successfully!")
except Exception as e:
    print(f"Training failed: {e}")
    import traceback
    traceback.print_exc()