from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

model_name = "trained_model"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def chat():
    print("Chatbot is ready! Type 'exit' to quit.")
    while True:
        question = input("You: ")
        if question.lower() == "exit":
            break
        inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=32)
        outputs = model.generate(inputs["input_ids"], max_length=32, num_beams=4)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Bot:", answer)
if __name__ == "__main__":
    chat()
