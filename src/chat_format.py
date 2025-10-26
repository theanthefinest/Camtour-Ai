# === dataset_prep.py ===

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from config import BASE_MODEL, DATA_PATH

# === Load Dataset ===
def load_and_prepare_dataset():
    dataset = load_dataset("json", data_files=DATA_PATH)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Format chat
    def chat_format(examples):
        return {"prompt": tokenizer.apply_chat_template(examples["messages"], tokenize=False)}

    formatted_dataset = dataset.map(chat_format)

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(examples["prompt"], truncation=True, padding=True, max_length=1024)

    tokenized_dataset = formatted_dataset.map(tokenize_function, batched=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    return tokenized_dataset, data_collator, tokenizer
