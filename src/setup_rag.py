# -------------------------------------------------
# ------------------ SETUP RAG --------------------
# -------------------------------------------------

'''
Script prepares the building of a RAG able of taking queries.

It requires an API key specified in file on path set below.

It also requires data prepared, currently using MIT AI NEWS.

'''

# ------------------ SET PATHS --------------------
# Root path to repo (when using ucloud, put /work/YourFolder#XXXX/repoName
path_to_root = '/work/PernilleHøjlundBrams#8577/NLP_2023_P'

# To API key file
path_to_key = f'{path_to_root}/config/keys.txt'

# To your data folder
path_to_data = f'{path_to_root}/data'

# To where you have your vector index stored (pre-made embeddings)
path_to_vector_store = f'{path_to_root}/index'

def do_setup():
        
    # ------------------ GET DATA --------------------
    # import pandas as pd

    # df = pd.read_csv(f"{path_to_data}/articles.csv", sep = ",").drop(columns = ["Unnamed: 0"])

    # # Msg
    # if df.empty:
    #     print("Data could not be loaded into dataframe (empty df).")
    # else: 
    #     print(f"Data loaded into dataframe. Data has length of {len(df)}")

    # # ------------------ PREP DATA --------------------
    # # Convert the DataFrame into a list of Document objects that the index can understand
    # from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, Document

    # documents = [Document(text=row['Article Body'],
    #                     metadata={'title': row['Article Header'],
    #                                 'source': row['Source'],
    #                                 'author': row['Author'],
    #                                 'date': row['Published Date'],
    #                                 'url': row['Url']}) for index, row in df.iterrows()] 

    # # Message
    # if documents:
    #     print("Documents made.")
    # else: 
    #     print("Documents could not be made.")

    # ------------------ SET API KEY --------------------
    from utils import read_api_key

    # Read and set API key
    api_key = read_api_key(path_to_key)

    if api_key:
        print("API key found. Setting as environmental variable.")

        import os
        # Set the OpenAI API key in the environment variables
        os.environ["OPENAI_API_KEY"] = api_key

        # Use the API key in your script
    else:
        print("No API key found.")

    # ------------------ LOAD LLM --------------------
    import torch
    from transformers import BitsAndBytesConfig
    from llama_index.prompts import PromptTemplate
    from llama_index.llms import HuggingFaceLLM

    # Msg
    print("Starting to load LLM.")

    # -- Load Mistral 7B Instruct as your llm
    llm = HuggingFaceLLM(
        model_name="mistralai/Mistral-7B-Instruct-v0.1",
        tokenizer_name="mistralai/Mistral-7B-Instruct-v0.1",
        query_wrapper_prompt=PromptTemplate("<s>[INST] {query_str} [/INST] </s>\n"),
        context_window=6000, # 3900 default
        max_new_tokens=350, # 256 default
        # tokenizer_kwargs={},
        #generate_kwargs={"temperature": 0.2, "top_k": 5, "top_p": 0.95},
        device_map="auto",
    )

    # Msg
    if llm:
        print("Finished loading LLM. LLM was loaded succesfully.")
    else: 
        print("Could not finish loading LLM.")

    # ------------------ IMPORT LLAMAINDEX STUFF -------
    # Msg
    print("Preparing service context.")

    from llama_index import (
        ServiceContext,
        OpenAIEmbedding,
        PromptHelper,
    )

    # --------- SPLIT IN CHUNKS and SET PROMPT HELPER -------
    from llama_index.text_splitter import SentenceSplitter

    #text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
    text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=10)

    prompt_helper = PromptHelper(
        context_window=6000,
        num_output=350,
        chunk_overlap_ratio=0.1,
        chunk_size_limit=None,
    )

    # ------------------ GET EMBEDDING MODEL ----------------
    embed_model = OpenAIEmbedding()

    # ------------------ SET SERVICE CONTEXT ----------------
    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        text_splitter=text_splitter,
        prompt_helper=prompt_helper,
    )

    # --------- GET VECTOR INDEX FROM STORAGE --------------
    import llama_index
    from llama_index import StorageContext, load_index_from_storage

    # Msg
    print("Loading in vector index from storage.")

    # Rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir=f"{path_to_vector_store}/df_aug_absa_nodes_index") # full_dataset_index, full_dataset_nodes_index

    # Load index from the storage context
    index = load_index_from_storage(storage_context)

    # Msg
    if index:
        print("Finished loading in vector storage. Index was loaded succesfully.")
    else: 
        print("Could not finish loading in vector storage.")
    
    print("-----------------------------")
    print("Setup done.")
    return index, llm # removed df,documents