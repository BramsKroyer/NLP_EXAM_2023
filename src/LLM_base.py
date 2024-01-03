# -------------------------------------------------
# ---------------- BUILD LLM BASE -----------------
# -------------------------------------------------

# Load Mistral 7B Instruct
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

# Define function for Mistral querying
def ask_LLM(query_str):
    model_inputs = tokenizer([query_str], return_tensors="pt")
     
    # Generate response
    generated_ids = model.generate(**model_inputs, max_new_tokens=350, do_sample=True)
    
    # Decode generated text
    full_text = tokenizer.batch_decode(generated_ids)[0] # this was 'answer' before

    # Remove the input part from the output (otherwise it prints the question asked)
    answer = full_text[len(tokenizer.decode(model_inputs["input_ids"][0])):].strip()

    print(answer)
    return answer