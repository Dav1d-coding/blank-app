from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("sarvamai/sarvam-1")
tokenizer = AutoTokenizer.from_pretrained("sarvamai/sarvam-1")

# Example usage
text = "hello freak"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=5)
result = tokenizer.decode(outputs[0])
