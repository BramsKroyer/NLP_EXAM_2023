# -------------------------------------------------
# -------------------- UTILS ----------------------
# -------------------------------------------------

# Function to format a single metadata dictionary into a string
def format_metadata(metadata):
    # Create a list of strings, each in the format "Key: Value"
    metadata_str_list = [f"{key}: {value}" for key, value in metadata.items()]
    # Join the list into a single string with line breaks
    return "\n".join(metadata_str_list)


# Generate response based on prompt
def generate_response(retrieved_nodes, query_str, qa_prompt, llm, metadata_list):
    
    context_str = "\n\n".join([r.get_content() for r in retrieved_nodes])
    meta_str = "\n\n".join([format_metadata(metadata) for metadata in metadata_list])
    
    fmt_qa_prompt = qa_prompt.format(
        meta_str=meta_str, context_str=context_str, query_str=query_str
    )
    response = llm.complete(fmt_qa_prompt)
    return str(response), fmt_qa_prompt

# Function to read an api key. If found, it is set as environment variable
def read_api_key(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        print("API key file not found.")
        return None

# Function to extraxt text snippet from node columns
def extract_text_snippet(text):
    import re

    # Adjust the regular expression to match text from 'Text:' up to '...'
    match = re.search(r'Text:\s*(.*?)\s*(?:\.\.\.|$)', text, re.DOTALL)
    if match:
        # Extract up to the first 10-15 tokens
        tokens = match.group(1).replace('\n', ' ').split()
        return ' '.join(tokens[:15]) if len(tokens) >= 10 else ' '.join(tokens)
    else:
        return None

