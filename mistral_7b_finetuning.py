import json
import os

import pandas as pd
import torch
import wandb
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset, Dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, setup_chat_format


with open("all_data.json") as f:
    dataset = json.load(f)
    df = pd.DataFrame(dataset)
    hf_dataset = Dataset.from_pandas(df)
    hf_dataset.save_to_disk('dataset_logical_reasoning')

# Set torch dtype and attention implementation
torch_dtype = torch.float16
attn_implementation = "eager"


# LoRA config based on QLoRA paper
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['o_proj', 'v_proj', 'gate_proj', 'k_proj', 'q_proj', 'down_proj', 'up_proj']
)



model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.3",

                                             attn_implementation=attn_implementation
                                             )
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")

model, tokenizer = setup_chat_format(model, tokenizer)

model.config.use_cache = False  # silence the warnings
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()

tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True

# prepare model for training
model = get_peft_model(model, peft_config)

args = TrainingArguments(
    output_dir="./mistral_7b_finetuning",
    num_train_epochs=4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    learning_rate=1e-7,
    max_grad_norm=0.3,
    warmup_ratio=0.06,
    lr_scheduler_type="cosine",
    logging_steps=2,
    logging_first_step=True,
    save_strategy="steps",
    save_steps=4,
    evaluation_strategy="no",
    eval_steps=20,
    disable_tqdm=False
)


def formatting_prompts_func(dataset_entry):
    output_texts = []
    for i in range(len(dataset_entry['problem_description'])):
        if dataset_entry['additional_problem_info'] != "":
            text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{dataset_entry['problem_description'][i]}\n\n{dataset_entry['additional_problem_info'][i]}<|im_end|>\n<|im_start|>assistant\n{dataset_entry['chain_of_thought'][i]}\n\n{dataset_entry['correct_solution'][i]}<|im_end|>"
        else:
            text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{dataset_entry['problem_description'][i]}<|im_end|>\n<|im_start|>assistant\n{dataset_entry['chain_of_thought'][i]}\n\n{dataset_entry['correct_solution'][i]}<|im_end|>"
        output_texts.append(text)
    return output_texts


# set the wandb project where this run will be logged
os.environ["WANDB_PROJECT"] = "BrainGPT"

response_template = "<|im_start|>assistant"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

trainer = SFTTrainer(
    model.to("cuda"),
    max_seq_length=4096,
    tokenizer=tokenizer,
    peft_config=peft_config,
    train_dataset=hf_dataset,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    args=args,

)

trainer.train()
trainer.save_model("./result_model")
