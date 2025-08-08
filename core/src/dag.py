from .tools import Tool, Dummy1to1Tool
from typing import Any, Callable, Tuple, Union

from dataclasses import dataclass


class Node:
    node_type: str = "Node"

    def __init__(self, tool: Tool) -> None:
        self.tool = tool

    def compute(self, inputs: list) -> Any:
        self.tool.validate_args(*inputs)
        out = self.tool.func(*inputs)

        if self.tool.num_outputs > 1:
            return list(out)
        else:
            return [out]



class InOutNode(Node):
    def __init__(self) -> None:
        super().__init__(tool=Dummy1to1Tool())

    def compute(self, inputs: list) -> Any:
        return inputs


class OutputNode(InOutNode):
    node_type = "OutputNode"


class InputNode(InOutNode):
    node_type = "InputNode"

    def __init__(self) -> None:
        super().__init__()
        self.data = None

    def populate(self, data: list[Any]) -> None:
        self.data = data

    def is_populated(self) -> bool:
        return self.data is not None


@dataclass
class Edge:
    src_node: Node
    tgt_node: Node
    src_out_arg_idx: int = 0
    tgt_in_arg_idx: int = 0


class Graph:
    def __init__(self) -> None:
        self.nodes: list[Node] = []
        self.edges: list[Edge] = []

        self.output_nodes: list[OutputNode] = []
        self.input_nodes: list[InputNode] = []

        self.node_outputs: dict[Node, list] = {}

    def add_node(self, func: Callable) -> Node:
        tool = Tool(func)
        node = Node(tool)
        self.nodes.append(node)
        return node

    def register_output_node(self) -> OutputNode:
        output_node = OutputNode()
        self.output_nodes.append(output_node)
        return output_node

    def get_output_node(self, idx: int) -> Node:
        assert 0 <= idx < len(self.output_nodes), "Invalid output node index"
        return self.output_nodes[idx]

    def register_input_node(self) -> InputNode:
        input_node = InputNode()
        self.input_nodes.append(input_node)
        return input_node

    def get_input_node(self, idx: int) -> Node:
        assert 0 <= idx < len(self.input_nodes), "Invalid input node index"
        return self.input_nodes[idx]

    def populate_inputs(self, input_data_dict: dict[InputNode, list]) -> None:
        for input_node, data in input_data_dict.items():
            assert input_node in self.input_nodes, "Invalid input node"
            input_node.populate(data)

    def add_edge(
        self,
        from_node: Union[Node, Tool],
        to_node: Union[Node, Tool],
        from_out_arg_idx: int = 0,
        to_in_arg_idx: int = 0,
    ) -> None:

        if isinstance(from_node, Tool):
            node = next(filter(lambda n: n.tool == from_node, self.nodes))
            assert node is not None, f"Node {from_node} not found in graph"
            from_node = node

        if isinstance(to_node, Tool):
            node = next(filter(lambda n: n.tool == to_node, self.nodes))
            assert node is not None, f"Node {to_node} not found in graph"
            to_node = node

        self.edges.append(Edge(from_node, to_node, from_out_arg_idx, to_in_arg_idx))

    def get_connected_edges(self, node: Node, sort: bool = False) -> list[Edge]:
        connected_edges = [edge for edge in self.edges if edge.tgt_node == node]

        if sort:
            connected_edges = sorted(
                connected_edges, key=lambda edge: edge.tgt_in_arg_idx
            )

        return connected_edges

    def fire(self) -> list[Any]:

        assert all(
            input_node.is_populated() for input_node in self.input_nodes
        ), "Input nodes not populated"

        outputs = []

        for output_node in self.output_nodes:
            outputs += self.compute_node(output_node)

        return outputs if len(outputs) > 1 else outputs[0]

    def compute_node(self, node: Node) -> Any:
        connected_edges = self.get_connected_edges(node, sort=True)

        inputs: list = []

        for connected_edge in connected_edges:
            src_node = connected_edge.src_node

            if isinstance(src_node, InputNode):
                assert src_node.data is not None, "Input node has not been populated"
                self.node_outputs[src_node] = src_node.data
            elif src_node not in self.node_outputs:
                self.node_outputs[src_node] = self.compute_node(src_node)

            inputs.append(self.node_outputs[src_node][connected_edge.src_out_arg_idx])

        return node.compute(inputs)
    
    def check_graph(self) -> Tuple[bool, str]:
        """
        Checks if the graph connections are valid based on input/output types.

        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        for edge in self.edges:
            src_node = edge.src_node
            tgt_node = edge.tgt_node

            # Skip type checking for InputNode and OutputNode pairs since they're pass-through
            if (isinstance(src_node, InputNode) and isinstance(tgt_node, OutputNode)) or \
            (isinstance(src_node, InputNode) or isinstance(tgt_node, OutputNode)):
                continue

            # Skip type checking if either node uses Dummy1to1Tool
            if isinstance(src_node.tool, Dummy1to1Tool) or isinstance(tgt_node.tool, Dummy1to1Tool):
                continue

            try:
                # Get output type from source node
                src_output_types = getattr(src_node.tool, 'output_types', None)
                if src_output_types is None:
                    continue
                if not isinstance(src_output_types, (list, tuple)):
                    src_output_types = [src_output_types]

                # Get input type from target node
                tgt_input_types = getattr(tgt_node.tool, 'input_types', None)
                if tgt_input_types is None:
                    continue
                if not isinstance(tgt_input_types, (list, tuple)):
                    tgt_input_types = [tgt_input_types]

                # Check if indices are valid
                if edge.src_out_arg_idx >= len(src_output_types):
                    return False, f"Invalid source output index {edge.src_out_arg_idx} for node {getattr(src_node.tool, 'name', 'Unknown')}"

                if edge.tgt_in_arg_idx >= len(tgt_input_types):
                    return False, f"Invalid target input index {edge.tgt_in_arg_idx} for node {getattr(tgt_node.tool, 'name', 'Unknown')}"

                # Get the specific types we're connecting
                src_type = src_output_types[edge.src_out_arg_idx]
                tgt_type = tgt_input_types[edge.tgt_in_arg_idx]

                # Check if types are compatible
                if not issubclass(src_type, tgt_type):
                    return False, (f"Type mismatch in connection: "
                                f"{getattr(src_node.tool, 'name', 'Unknown')}[{edge.src_out_arg_idx}] ({src_type.__name__}) -> "
                                f"{getattr(tgt_node.tool, 'name', 'Unknown')}[{edge.tgt_in_arg_idx}] ({tgt_type.__name__})")

            except AttributeError:
                # If type information is not available, skip type checking for this edge
                continue

        # Check for cycles
        visited = set()
        temp_visited = set()

        def has_cycle(node: Node) -> bool:
            if node in temp_visited:
                return True
            if node in visited:
                return False

            temp_visited.add(node)

            # Get all nodes this node connects to
            for edge in self.edges:
                if edge.src_node == node:
                    if has_cycle(edge.tgt_node):
                        return True

            temp_visited.remove(node)
            visited.add(node)
            return False

        # Check each node for cycles
        for node in self.nodes:
            if node not in visited:
                if has_cycle(node):
                    return False, "Graph contains cycles"

        return True, "Graph is valid"