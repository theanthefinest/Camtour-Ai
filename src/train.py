# === train.py ===

from transformers import Trainer
from config import training_args
from chat_format import load_and_prepare_dataset
from model_setup import setup_model
from utils import hf_login, print_summary

if __name__ == "__main__":
    try:
        hf_login()

        print("📚 Loading dataset...")
        tokenized_dataset, data_collator, tokenizer = load_and_prepare_dataset()

        print("🧠 Setting up model...")
        model = setup_model(new_lora=True)

        print_summary(model, tokenized_dataset)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            data_collator=data_collator,
        )

        print("🚀 Starting training...")
        trainer.train()
        print("✅ Training completed successfully!")

    except Exception as e:
        import traceback
        print(f"❌ Training failed: {e}")
        traceback.print_exc()
