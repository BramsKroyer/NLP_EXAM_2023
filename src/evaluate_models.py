# -------------------------------------------------
# ------------ EVALUATE MODELS --------------------
# -------------------------------------------------

# ---------------- SET API KEY --------------------
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

# ------------ DEFINE SERVICE CONTEXT --------------
from llama_index.llms import OpenAI
from llama_index import ServiceContext
service_context = ServiceContext.from_defaults(llm=OpenAI("gpt-3.5-turbo-0301"))

# -------------------------------------------------
# ***** TESTS FOR BOTH LLM and LLM+RAG ************
# -------------------------------------------------

# ------------ EVALUATE CORRECTNESS ---------------
from llama_index.evaluation import CorrectnessEvaluator

# -- Define evaluator
correctness_evaluator = CorrectnessEvaluator(service_context=service_context)

# -- Define function for correctness evaluation
def correctness_eval(query_str, response_str, reference_str):
    
    result = correctness_evaluator.evaluate(
    query=query_str,
    response=response_str,
    reference=reference_str,
    )

    return result.passing, result.score, result.feedback

# -------- EVALUATE SEMANTIC SIMILARITY (between response and gold response or between query and response) -------------
# (a high score does not imply the answer is always correct - but whether the things talked about are semantically similar)
from llama_index.evaluation import SemanticSimilarityEvaluator

# -- Define evaluator
sem_sim_evaluator = SemanticSimilarityEvaluator(service_context=service_context) 

# -- Define function for semantic similarity evaluation
def semsim_eval(response_or_query_str, reference_str):
    
    result = sem_sim_evaluator.evaluate(
    # Can be both response or query_str in theory
    response=response_or_query_str,
    reference=reference_str,
    )

    return result.passing, result.score

# -------------------------------------------------
# ******* TESTS FOR LLM+RAG ONLY ******************
# -------------------------------------------------

# ------------- EVALUATE RELEVANCY ----------------
from llama_index.evaluation import RelevancyEvaluator

# -- Define evaluator
relevancy_evaluator = RelevancyEvaluator(service_context=service_context)

# -- Define function for relevancy evaluation
def relevancy_eval(query_str, response_str, retrieved_nodes):

    # source_nodes = list()
    # for i in retrieved_nodes:
    #     i = str(i)
    #     source_nodes.append(i)
    print(f"retrieved_nodes is currently of length: {len(retrieved_nodes)} and of type {type(retrieved_nodes)} and is: {retrieved_nodes}")
    result = relevancy_evaluator.evaluate(query=query_str,response=response_str, contexts=retrieved_nodes)#contexts=source_nodes)

    return result.passing, result.score, result.feedback

# ------------ EVALUATE FAITHFULLNESS -------------
from llama_index.evaluation import FaithfulnessEvaluator

# -- Define evaluator
faithful_evaluator = FaithfulnessEvaluator(service_context=service_context)

# -- Define function for faithfullness evaluation
def faithfullness_eval(response_str, retrieved_nodes):
    
    # source_nodes = list()
    # for i in retrieved_nodes:
    #     i = str(i)
    #     source_nodes.append(i)
    print(f"retrieved_nodes is currently of length: {len(retrieved_nodes)} and of type {type(retrieved_nodes)} and is: {retrieved_nodes}")
    result = faithful_evaluator.evaluate(response=response_str,contexts=retrieved_nodes)

    return result.passing, result.score, result.feedback


