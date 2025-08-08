from typing import Dict, Any, Optional
import json
from datetime import datetime



class GraphStateManager:
    def __init__(self):
        self.history = []
        self.failed_attempts = []

    def add_attempt(self, query: str, config: Dict, inputs: Dict, success: bool, error_message: Optional[str] = None):
        attempt = {
            "query": query,
            "config": config,
            "inputs": inputs,
            "success": success,
            "error_message": error_message,
            "timestamp": datetime.now().isoformat()
        }
        self.history.append(attempt)
        if not success:
            self.failed_attempts.append(attempt)

    def get_context_for_retry(self) -> str:
        """Generate context from previous attempts for the LLM."""
        context = []
        if self.failed_attempts:
            context.append("Previous failed attempts:")
            for attempt in self.failed_attempts[-3:]:  # Last 3 failures
                context.append(f"Query: {attempt['query']}")
                context.append(f"Error: {attempt['error_message']}")
                context.append(f"Config tried: {json.dumps(attempt['config'], indent=2)}")
                context.append(f"Documented attempt time: {attempt['timestamp']}")
                context.append("---")
        return "\n".join(context)

def get_corrected_dag_config(query: str, original_config: Dict, error_message: str, state_manager: GraphStateManager, llm, debug: bool = False) -> Dict:
    """Ask LLM to correct the DAG configuration based on the error."""
    # take a prompt to be more exact
    with open("system_prompts/dag_rewrite_structured.txt") as f:
        q = f.read() + "\n" + query
    prompt = f"""
    The previous system prompt and examples:{q}
    The following DAG configuration failed validation:

    Original Query: {query}
    Configuration: {json.dumps(original_config, indent=2)}
    Error: {error_message}

    Previous attempts context:
    {state_manager.get_context_for_retry()}

    Please provide a corrected DAG configuration that:
    1. Addresses the specific validation error
    2. Maintains the original query's intent
    3. Ensures type compatibility between connected nodes
    4. Avoids cycles in the graph

    Respond with only the corrected JSON configuration.
    """
    if debug:
        print("In debug mode!!!",prompt)
        return ("debug", prompt)
    else:
    # Get corrected configuration from LLM
        corrected_config = llm.generate(prompt)
        return json.loads(corrected_config)

