
"""agent_scaffold.py

This module provides the AgentScaffold class for managing LLM interaction:
- Prompt-based query rewriting
- Output validation and JSON self-correction
- Basic chat handling

"""

import json
import re

class agentscaffold:
    def __init__(self, query: str, llm):
        self.query = query
        self.llm = llm

    def rewrite_query(self, query, prompt_file="system_prompts/initial_rewrite.txt") -> str:
        """Rewrites the user query using a system prompt."""
        with open(prompt_file) as f:
            prompt = f.read() + "\n" + str(query)
        return self.llm.complete(prompt).text

    def check_output_parsability(self, input_response: str) -> tuple:
        """Checks if the response contains valid JSON after removing formatting tags."""
        try:
            input_response = re.sub(r"<think>.*?</think>", "", input_response, flags=re.DOTALL)

            if input_response.count("**JSON**") != 2:
                raise Exception("Missing **JSON** tags.")

            cleaned_response = (
                input_response.replace("**JSON**", "")
                .replace("```json", "")
                .replace("```", "")
                .replace("json", "")
                .replace("\n", "")
                .strip()
            )

            json.loads(cleaned_response)  # validate
            return (None, True, cleaned_response)

        except Exception as e:
            return (str(e), False, None)

    def self_correction(self, input_response: str, max_iterations: int = 10) -> str:
        """Iteratively correct the output until it's parsable JSON or max attempts are reached."""
        memory = []

        for iteration in range(max_iterations):
            error, success, cleaned_response = self.check_output_parsability(input_response)
            print(f"Iteration {iteration + 1}: {cleaned_response}")

            if success:
                return cleaned_response

            memory.append(error)
            error_context = " | ".join(memory)

            with open("system_prompts/self_correction_system_prompts.txt") as f:
                prompt = (
                    f.read() +
                    f"\nErrors so far: {error_context}" +
                    "\nInput response:\n" + input_response
                )
            input_response = self.get_llm_response(prompt)

        print("Max iterations reached. Returning last attempted response.")
        return input_response

    def get_llm_response(self, query: str) -> str:
        """Sends a raw prompt to the LLM."""
        return self.llm.complete(query).text
