from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Step 1: Load the GPT-2 model and tokenizer
model_name = "gpt2"  # You can choose from 'gpt2', 'gpt2-medium', 'gpt2-large', etc.
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Step 2: Set the pad_token to eos_token or add a custom padding token
tokenizer.pad_token = tokenizer.eos_token
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Alternative option

# Step 3: Load and prepare your dataset
# Replace 'dataset.txt' with the path to your dataset
dataset = load_dataset('text', data_files={'train': 'dataset.txt'})

# Tokenize the dataset and set labels
def tokenize_function(examples):
    inputs = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)
    inputs["labels"] = inputs["input_ids"].copy()  # Set labels to be the same as input_ids
    return inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Step 4: Set up training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,  # Adjust the number of epochs as needed
    per_device_train_batch_size=2,  # Adjust batch size according to your GPU's capability
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=500,
)

# Step 5: Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
)

# Step 6: Fine-tune the model
trainer.train()

# Step 7: Save the fine-tuned model
trainer.save_model("./gpt2-finetuned")
tokenizer.save_pretrained("./gpt2-finetuned")

# Step 8: Generate text using the fine-tuned model
# Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained("./gpt2-finetuned")
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-finetuned")

# Generate text with the fine-tuned model
prompt = "In a distant future,"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(inputs['input_ids'], max_length=100, num_return_sequences=1)

# Decode and print the generated text
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
