# chatbot_flan_t5.py

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

def main():
    # Hugging Face model repo
    fine_tuned_model_name = "naidu9678/flan_t5_finetuned_education"

    # Load tokenizer and model from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(fine_tuned_model_name)

    # Create text2text-generation pipeline
    device = 0 if torch.cuda.is_available() else -1
    chatbot = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)

    # Example question and context
    question = "What was Naidu's GPA ?"
    context = "Naidu is a senior Cloud Computing student at BITS Pilani. He has a GPA of 8.4 and is part of the AI research group."

    input_text = f"question: {question} context: {context}"

    # Generate answer
    outputs = chatbot(
        input_text,
        max_new_tokens=50,
        do_sample=True,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=3,
        temperature=0.7
    )

    print(f"Question: {question}")
    print(f"Answer: {outputs[0]['generated_text']}")

if __name__ == "__main__":
    main()
