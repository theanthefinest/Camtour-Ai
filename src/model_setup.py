# === model_setup.py ===

import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from config import BASE_MODEL, bnb_config, RESUME_MODEL_PATH

def setup_model(new_lora=True):
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # Resume from existing LoRA if available
    try:
        temp_model = PeftModel.from_pretrained(model, RESUME_MODEL_PATH)
        model = temp_model.merge_and_unload()
        print(f"Resumed model from: {RESUME_MODEL_PATH}")
    except Exception:
        print("No previous LoRA checkpoint found. Starting fresh.")

    # Prepare model
    model = prepare_model_for_kbit_training(model)

    if new_lora:
        new_lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, new_lora_config)
        print("✅ Applied new LoRA configuration.")

    model.train()
    model.config.use_cache = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")
    return model
