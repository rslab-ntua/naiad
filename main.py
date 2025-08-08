# main.py

"""
Command-line entry point for the NAIAD agentic RAG system.

This script initializes the LLM, loads the main agent,
and allows interactive querying via the terminal.
"""

from llama_index.llms.ollama import Ollama
from main_agent import agent

# Configure LLM instance (adjust base_url if needed)
llm = Ollama(
    model="qwen2.5:14b",
    base_url="http://localhost:11434",
    request_timeout=120.0,
    temperature=0.0,
)

def main():
    print("NAIAD Interactive Agent")
    print("Type a natural language query or 'exit' to quit.")
    while True:
        query = input("\nEnter your query: ")
        if query.strip().lower() == "exit":
            break
        if not query.strip():
            continue

        try:
            agent_instance = agent(query, llm)
            result = agent_instance.run()
            print("\n--- Response ---")
            print(result)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
