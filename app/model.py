from transformers import GPT2LMHeadModel, GPT2Tokenizer


# Загрузка модели и токенайзера
def load_model():
    model_name = "datificate/gpt2-small-spanish"  # GPT-2 на испанском
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return tokenizer, model


# Генерация текста
def generate_response(prompt: str, tokenizer, model, max_length=100):
    inputs = tokenizer.encode(prompt, return_tensors="pt", padding=True, truncation=True)
    attention_mask = inputs != tokenizer.pad_token_id
    outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,  # Включаем выборку для использования temperature
        top_p=0.9,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id  # Устанавливаем паддинг
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
