
"""main_agent.py

Entry point for the NAIAD agent system. It wraps:
- DAG construction
- Input extraction
- Tool execution
- Relevancy feedback loop

"""

import json
from agent_controller import (
    build_graph_from_json,
    get_node_input_from_query,
    relevancy_check,
    key_provider,
    is_general_report_query
)
from agent_scaffold import agentscaffold
from src.state import GraphStateManager, get_corrected_dag_config
from context_utils import original_query_var

class agent:
    def __init__(self, query: str, llm):
        self.query = query
        self.llm = llm
        self.scaffold = agentscaffold(query, llm)
        self.state_manager = GraphStateManager()

    def build_dag_config(self, inputs_dict=None):
        """Decides DAG structure based on the query and inputs."""
        if is_general_report_query(self.query, self.llm):
            print("Using shortcut: general report query detected.")
            if not inputs_dict:
                return {
                    "nodes": {
                        "input0": {"type": "inputNode"},
                        "output": {"type": "outputNode"},
                        "rag_report_general_node": {"type": "Node", "function": "rag_report_general"},
                    },
                    "edges": [
                        {"from_node": "input0", "to_node": "rag_report_general_node", "from_idx": 0, "to_idx": 0},
                        {"from_node": "rag_report_general_node", "to_node": "output", "from_idx": 0, "to_idx": 0},
                    ],
                }

            # Multi-lake config
            nodes = {}
            edges = []
            for i, (input_key, input_value) in enumerate(inputs_dict.items()):
                nodes[f"input{i}"] = {"type": "inputNode"}
                nodes[f"rag_report_general_node_{input_value.lower()}"] = {"type": "Node", "function": "rag_report_general"}
                edges.append({
                    "from_node": f"input{i}",
                    "to_node": f"rag_report_general_node_{input_value.lower()}",
                    "from_idx": 0,
                    "to_idx": 0
                })

            nodes["merge_outputs_node"] = {"type": "Node", "function": "merge_outputs"}
            nodes["output"] = {"type": "outputNode"}
            for idx, (_, input_value) in enumerate(inputs_dict.items()):
                edges.append({
                    "from_node": f"rag_report_general_node_{input_value.lower()}",
                    "to_node": "merge_outputs_node",
                    "from_idx": 0,
                    "to_idx": idx
                })
            edges.append({
                "from_node": "merge_outputs_node",
                "to_node": "output",
                "from_idx": 0,
                "to_idx": 0
            })

            return {"nodes": nodes, "edges": edges}

        # General DAG from prompt
        extra_prompt = ""
        if inputs_dict:
            formatted_inputs = "\n".join([f"{k}: {v}" for k, v in inputs_dict.items()])
            extra_prompt = f"\n\nDetected inputs:\n{formatted_inputs}\nUse each input node explicitly."

        raw = self.scaffold.rewrite_query(
            self.query + extra_prompt,
            prompt_file="system_prompts/dag_rewrite_structured.txt"
        )
        return json.loads(self.scaffold.self_correction(raw))

    def build_graph(self, config):
        return build_graph_from_json(config, llm=self.llm)

    def get_inputs(self, config):
        return get_node_input_from_query(self.query, config, llm=self.llm)

    def validate_graph(self, graph):
        return graph.check_graph()

    def execute_graph(self, graph, inputs, nodes):
        for key, value in inputs.items():
            graph.populate_inputs({nodes[key]: [value]})
        return graph.fire()

    def check_relevancy(self, outputs):
        return relevancy_check(self.query, outputs, llm=self.llm)

    def run(self, max_retries=3, relevancy_retries=3):
        """Main execution loop for DAG-based agentic decision making."""
        original_query_var.set(self.query)
        current_try = 0

        lake_inputs = key_provider(self.query, self.llm)
        input_dict = {f"input{i}": lake for i, lake in enumerate(lake_inputs)}

        while current_try < max_retries:
            try:
                config = self.build_dag_config(inputs_dict=input_dict)
                inputs = input_dict

                graph, nodes = self.build_graph(config)
                is_valid, error_msg = self.validate_graph(graph)

                if not is_valid:
                    self.state_manager.add_attempt(self.query, config, inputs, False, error_msg)
                    if current_try < max_retries - 1:
                        config = get_corrected_dag_config(
                            query=self.query,
                            original_config=config,
                            error_message=error_msg,
                            state_manager=self.state_manager,
                            llm=self.llm
                        )
                        current_try += 1
                        continue
                    break

                outputs = self.execute_graph(graph, inputs, nodes)

                for rel_try in range(relevancy_retries):
                    rel_result = self.check_relevancy(outputs).strip().upper()
                    if rel_result == "YES":
                        self.state_manager.add_attempt(self.query, config, inputs, True)
                        return outputs
                    else:
                        config = self.build_dag_config()
                        inputs = self.get_inputs(config)
                        graph, nodes = self.build_graph(config)
                        outputs = self.execute_graph(graph, inputs, nodes)
                break

            except Exception as e:
                self.state_manager.add_attempt(
                    self.query,
                    config if 'config' in locals() else {},
                    inputs if 'inputs' in locals() else {},
                    False,
                    str(e)
                )
                current_try += 1

        return None
