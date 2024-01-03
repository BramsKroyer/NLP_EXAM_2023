# -------------------------------------------------
# ------------------- BUILD RAG -------------------
# -------------------------------------------------

# Import setup
from setup_rag import do_setup

# This will run the setup before building the RAG
index, llm = do_setup() # removed df, documents,

from utils import format_metadata, generate_response

from llama_index.llms import OpenAI
from llama_index.prompts import PromptTemplate

# ------------------ SET API KEY AGAIN --------------------
from utils import read_api_key
from setup_rag import path_to_key

# Read and set API key
api_key = read_api_key(path_to_key)

if api_key:

    import os
    # Set the OpenAI API key in the environment variables
    os.environ["OPENAI_API_KEY"] = api_key

    # Use the API key in your script
else:
    print("No API key found.")

# ------------------ PROMPT TEMPLATE --------------------
# --- PROMPT1
# qa_prompt = PromptTemplate(
#     """\
# Context information is below.
# ---------------------
# {context_str}
# {meta_str}
# --------------------- 
# Answer the query.
# If you cannot find the answer in the given context, answer based on prior knowledge.
# If you answer based on prior knowledge, you must prefix your response with "Based on prior knowledge". Otherwise prefix your response with "Based on the context".
# In the end of the answer include a list with the URLs found in the metadata.

# Query: {query_str}
# Answer: \
# """
# )

# --- PROMPT2
# qa_prompt = PromptTemplate(
#     """\
# Context information is below.
# ---------------------
# {context_str}
# {meta_str}
# --------------------- 
# Answer the query.
# If you cannot find the answer in the given context, you must try to answer based on prior knowledge.
# If you answer based on prior knowledge, prefix your response with "Based on prior knowledge". Otherwise prefix your response with "Based on the context".

# Query: {query_str}
# Answer: \
# """
# )

# --- PROMPT 3
# qa_prompt = PromptTemplate(
#     """\
# Context information is below.
# ---------------------
# {context_str}
# {meta_str}
# --------------------- 
# If you do not find the answer in the given context, answer based on prior knowledge.
# If you answer based on prior knowledge, prefix your response with "Based on prior knowledge". Otherwise prefix your response with "Based on the context".

# Query: {query_str}
# Answer: \
# """
# )

# --- PROMPT 4
qa_prompt = PromptTemplate(
    """\
Context information is below.
---------------------
{context_str}
{meta_str}
--------------------- 
If you do not find the answer in the given context, answer based on prior knowledge.
If you answer based on prior knowledge, prefix your response with "Based on prior knowledge". Otherwise prefix your response with "Based on the context".
In the end of the answer include a list with the URLs found in the metadata.

Query: {query_str}
Answer: \
"""
)

# ------------------ PROMPT TEMPLATE --------------------
def RAG_mistral7B(query_str):

    # Retrieve nodes based on query
    print("Retrieving nodes...")
    retriever = index.as_retriever(similarity_top_k=3) #NOTE: 5 before
    retrieved_nodes = retriever.retrieve(query_str)

    # Get metadata list
    metadata_list = []

    for i in range(len(retrieved_nodes)):
        node_ = retrieved_nodes[i].node.metadata
        metadata_list.append(node_)

    # Generating response
    print("Preparing response...")

    response, fmt_qa_prompt = generate_response(
        retrieved_nodes, query_str, qa_prompt, llm, metadata_list
    )
    print(f"*****Response******:\n{response}\n\n")
    print(f"*****Formatted Prompt*****:\n{fmt_qa_prompt}\n\n")
    
    return str(response), fmt_qa_prompt, retrieved_nodes # Note, added this

print("RAG boi BUILT!!")
