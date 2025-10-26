# === config.py ===

from transformers import BitsAndBytesConfig, TrainingArguments

# === Hugging Face Authentication ===
HF_TOKEN = "your_huggingface_token"

# === Model and Dataset Paths ===
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
DATA_PATH = "your_data_path"
RESUME_MODEL_PATH = "your_model_path_checkpoint"
OUTPUT_DIR = "/content/drive/MyDrive/Colab Notebooks/Model/chatbot_v3"

# === Bits and Bytes (4-bit quantization) ===
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16"
)

# === Training Arguments ===
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    learning_rate=1.5e-4,
    num_train_epochs=3,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    optim="paged_adamw_8bit",
    warmup_ratio=0.1,
    report_to=None,
    gradient_checkpointing=True,
    max_grad_norm=0.3,
    lr_scheduler_type="cosine",
    logging_first_step=True,
    seed=42,
    dataloader_pin_memory=True,
    remove_unused_columns=True,
    save_safetensors=True,
)
