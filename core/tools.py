
"""tools.py

This module contains simplified tool interfaces for the NAIAD system, including:
- Weather querying and reasoning
- NDCI and Chlorophyll estimation
- Cyanobacteria level prediction using CyFi
- RAG-based general report generation
- Output merging logic

Note: All tools are examples or partial implementations for demonstration.
Full integrations, credentials, or APIs are omitted or stubbed.
"""

import json
import datetime
import numpy as np
import requests
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from context_utils import inject_original_query
from agent_scaffold import agentscaffold

@inject_original_query
def get_weather(query: str, llm, input_dir_path = './weather_rag_docs', original_query = None) -> str:
    """Uses lat/lon from LLM and calls Open-Meteo, then runs a RAG response generation."""
    full_context_query = f"{original_query} (target location: {query})" if original_query else query
    with open("system_prompts/weather_tool_prompt.txt") as f:
        q = f.read() + "\n" + full_context_query

    response_llm = llm.complete(q)
    lat_long = json.loads(response_llm.text)
    url = f'https://api.open-meteo.com/v1/forecast?latitude={lat_long["lat"]}&longitude={lat_long["long"]}&current=temperature_2m'

    try:
        response = requests.get(url)
        response.raise_for_status()
        weather_data = response.json()
    except Exception as e:
        print(f"Error while fetching weather data: {e}")
        weather_data = {}

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", trust_remote_code=True)
    Settings.embed_model = embed_model
    Settings.llm = llm

    docs = SimpleDirectoryReader(input_dir=input_dir_path, required_exts=[".txt"]).load_data()
    index = VectorStoreIndex.from_documents(docs)
    query_engine = index.as_query_engine(similarity_top_k=10)

    with open("system_prompts/weather_plus_rag_system_prompt.txt") as f:
        system_prompt = f.read()

    full_prompt = f"{system_prompt}\nUser input --> {full_context_query}\nWeather data --> {json.dumps(weather_data)}"
    return query_engine.query(full_prompt)

def NDCIfetchingFunction():
    """Returns a JSON with NDCI values for known lakes (stubbed)."""
    return {"mornos": 0.45, "trichonida": 0.33, "lisimacheia": 0.52}

def fetch_ndci_data(lake_name: str) -> float:
    """Returns the NDCI value for a given lake name."""
    ndci_values = NDCIfetchingFunction()
    return ndci_values.get(lake_name, 0.0)

def CalculatorChlorophyl(ndci: float) -> float:
    """Calculates Chlorophyl using NDCI."""
    return 5.441 * np.exp(7.29 * ndci) - 3

@inject_original_query
def get_cyanobacteria_levels_query_response(query: str, llm, input_dir_path = './cyfi_tool_rag_docs', original_query = None) -> str:
    """Calls CyFi API and combines prediction with a RAG-based explanation."""
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    full_query_context = f"{original_query} (target lake: {query})" if original_query else query

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", trust_remote_code=True)
    Settings.embed_model = embed_model
    Settings.llm = llm

    docs = SimpleDirectoryReader(input_dir=input_dir_path, required_exts=[".txt"]).load_data()
    index = VectorStoreIndex.from_documents(docs)
    query_engine = index.as_query_engine(similarity_top_k=10)

    query_with_date = f"{full_query_context} {current_date}"
    scaffold = agentscaffold(query_with_date, llm)
    raw = scaffold.rewrite_query(query_with_date)
    rewriten_query = scaffold.self_correction(raw)

    # Simulated output from a CyFi API call (replace with real call if desired)
    cyfi_json = {"prediction": "moderate", "risk": "elevated", "date": current_date}
    response = query_engine.query(rewriten_query + str(cyfi_json))
    return response

@inject_original_query
def rag_report_general(query:str, llm, input_path =['./rag_docs'], original_query=None)-> str:
    """Performs RAG across multi-folder docs to answer high-level questions."""
    full_context_query = f"{original_query} (focus: {query})" if original_query else query
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", trust_remote_code=True)
    Settings.embed_model = embed_model
    Settings.llm = llm

    all_docs = []
    for path in input_path:
        all_docs.extend(SimpleDirectoryReader(input_dir=path).load_data())

    index = VectorStoreIndex.from_documents(all_docs)
    query_engine = index.as_query_engine(similarity_top_k=10)
    return query_engine.query(full_context_query).response

@inject_original_query
def merge_outputs(*args, llm=None, original_query=None) -> str:
    """Uses the LLM to merge tool outputs into a human-readable final answer."""
    if llm is None:
        return str(args)

    cleaned_outputs = [getattr(item, "response", str(item)) for item in args]
    prompt = (
        "You are a smart assistant. The user asked a question. Each result below is the output from a tool, "
        "related to a specific location or item in the query. Merge them into a clear, user-friendly summary.\n\n"
        f"User Query:\n{original_query}\n\nResults:\n" + "\n".join(f"- {line}" for line in cleaned_outputs) + "\n\nFinal Answer:"
    )

    try:
        response = llm.complete(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[merge_outputs] LLM error: {e}")
        return "\n".join(cleaned_outputs)
