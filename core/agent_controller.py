
"""
agent_controller.py

This module demonstrates key components of the NAIAD system's agentic logic:
- Dynamic DAG construction from user queries
- Tool function mapping and LLM-wrapped execution
- Prompt-driven input prediction and validation
- RAG/report-only routing logic

Note: This is a simplified, illustrative version for publication purposes.
Actual tool implementations and LLM endpoints have been stubbed or omitted.
"""

import json
import re
from difflib import get_close_matches
from functools import wraps

from src.dag import Graph
from agent_scaffold import agentscaffold

# Dummy imports or stubs — replace with real tool implementations if available
from tools import (
    fetch_ndci_data,
    CalculatorChlorophyl,
    get_cyanobacteria_levels_query_response,
    get_weather,
    rag_report_general,
    downloader,
    merge_outputs
)

def build_graph_from_json(json_config: dict, llm) -> tuple[Graph, dict]:
    """Builds a graph from a JSON configuration using LLM-wrapped tools."""
    graph = Graph()
    nodes = {}

    def wrap_llm(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            return fn(*args, llm=llm, **kwargs)
        return wrapped

    function_map = {
        "CalculatorChlorophyl": CalculatorChlorophyl,
        "fetch_ndci_data": fetch_ndci_data,
        "get_cyanobacteria_levels_query_response": wrap_llm(get_cyanobacteria_levels_query_response),
        "get_weather": wrap_llm(get_weather),
        "rag_report_general": wrap_llm(rag_report_general),
        "downloader": wrap_llm(downloader),
        "merge_outputs": wrap_llm(merge_outputs)
    }

    for node_name, node_config in json_config["nodes"].items():
        if node_config["type"] == "inputNode":
            nodes[node_name] = graph.register_input_node()
        elif node_config["type"] == "outputNode":
            nodes[node_name] = graph.register_output_node()
        else:
            func = function_map[node_config["function"]]
            nodes[node_name] = graph.add_node(func)

    for edge in json_config["edges"]:
        graph.add_edge(
            nodes[edge["from_node"]], 
            nodes[edge["to_node"]], 
            edge["from_idx"], 
            edge["to_idx"]
        )

    return graph, nodes

def key_provider(query: str, llm, valid_lakes=["mornos", "trichonida", "lisimacheia"]) -> list[str]:
    """Extracts relevant lake names or keywords from a user query."""
    query_lower = query.lower()
    lake_keys = [lake for lake in valid_lakes if lake in query_lower]

    for word in re.findall(r'\b\w+\b', query_lower):
        match = get_close_matches(word, valid_lakes, n=1, cutoff=0.8)
        if match and match[0] not in lake_keys:
            lake_keys.append(match[0])

    if lake_keys:
        return lake_keys

    prompt = (
        "You are an expert assistant. Given a user query, extract the most relevant input value "
        "(either a number, term, or phrase) that the system should use. Return only the input value, no explanation.\n\n"
        f"Query: {query}\n\nInput:"
    )
    try:
        response = llm.complete(prompt)
        value = response.text.strip()
        return [value] if value else []
    except Exception as e:
        print(f"[key_provider] LLM fallback failed: {e}")
        return []

def is_general_report_query(query: str, llm) -> bool:
    """Determines if a query should be routed only to the general report tool."""
    prompt = f"""
You are an expert assistant.

Determine if the following user query should be handled ONLY by the 'rag_report_general' tool.
Answer YES if the query asks for:
- a general report,
- analytical or comparative reasoning,
- historical explanation,
- or insight based on existing documents.

Answer NO if it explicitly asks for:
- numerical values,
- measurements (e.g., current weather, NDCI),
- forecasts,
- real-time cyanobacteria predictions,
- or anything requiring external tool usage.

Respond with only YES or NO.

Query: {query}
Answer:"""
    try:
        response = llm.complete(prompt)
        return response.text.strip().upper().startswith("YES")
    except Exception as e:
        print(f"[is_general_report_query] LLM error: {e}")
        return False

def get_node_input_from_query(original_query, graph_config, llm):
    """Uses prompt-based reasoning to fill in DAG inputs from a query."""
    scaffold = agentscaffold(original_query, llm=llm)
    with open("system_prompts/dag_get_node_input_prompt.txt") as f:
        prompt_template = f.read()

    full_prompt = (
        prompt_template + "\nUser input--> " + original_query + "\nGraph --> " + str(graph_config)
    )

    raw = scaffold.get_llm_response(full_prompt)
    json_output = scaffold.self_correction(raw)
    structured_data = json.loads(json_output)

    for i, lake in enumerate(key_provider(original_query, llm)):
        structured_data[f'input{i}'] = lake

    return structured_data

def relevancy_check(original_query: str, response: any, llm):
    """Asks the LLM to judge whether the response is relevant to the original query."""
    with open("system_prompts/relevancy_judgement.txt") as f:
        q = f.read() + f"\nUser input--> {original_query}\nGenerated response --> {response}"
    try:
        relevancy_check = llm.complete(q)
        return relevancy_check.text
    except Exception as e:
        print(f"[relevancy_check] LLM error: {e}")
        return "RELEVANCY CHECK FAILED"
