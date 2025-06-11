import json
from transformers import Trainer, TrainingArguments, AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import Dataset


with open('company_data.json') as f:
    pairs = json.load(f)


train_data = {'input_text': [], 'target_text': []}
for item in pairs:
    train_data['input_text'].append(item['question'])
    train_data['target_text'].append(item['answer'])

dataset = Dataset.from_dict(train_data)

model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def preprocess(example):
    inputs = tokenizer(example['input_text'], truncation=True, padding='max_length', max_length=32)
    targets = tokenizer(example['target_text'], truncation=True, padding='max_length', max_length=32)
    inputs['labels'] = targets['input_ids']
    return inputs

tokenized_dataset = dataset.map(preprocess)


training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,  
    per_device_train_batch_size=2,
    logging_dir='./logs',
    logging_steps=10,
    save_steps=50,
    save_total_limit=2
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

trainer.train()
model.save_pretrained("trained_model")
tokenizer.save_pretrained("trained_model")
