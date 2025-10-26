# === utils.py ===

from huggingface_hub import login
from config import HF_TOKEN

def hf_login():
    print("🔐 Logging in to Hugging Face...")
    login(token=HF_TOKEN)
    print("✅ Login successful!")

def print_summary(model, tokenized_dataset):
    print("=== Final Verification ===")
    print(f"Model training mode: {model.training}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Dataset size: {len(tokenized_dataset['train'])}")
