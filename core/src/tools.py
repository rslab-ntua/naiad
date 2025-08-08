from typing import Callable


from .utils import Arg, func_input_args, func_output_args

TOOL_DESCRIPTION = """
Tool name: {name}, 
Tool number of inputs: {num_args},
Tool number of outputs: {num_outputs},
Description: {desc}
"""


class Tool:
    def __init__(self, func: Callable) -> None:
        super().__init__()
        assert func.__doc__, "Tool must have a docstring"
        self.func = func

    @property
    def num_args(self) -> int:
        return len(self.input_args)

    @property
    def input_args(self) -> list[Arg]:
        return func_input_args(self.func)

    def validate_args(self, *args) -> bool:
        if len(args) != self.num_args:
            return False
        #  Type checking
        return all(
            isinstance(input_arg, arg_definition.type)
            for input_arg, arg_definition in zip(args, self.input_args)
        )

    @property
    def num_outputs(self) -> int:
        return len(func_output_args(self.func))

    @property
    def name(self) -> str:
        return self.func.__name__

    def __str__(self) -> str:
        return TOOL_DESCRIPTION.format(
            name=self.name,
            num_args=self.num_args,
            num_outputs=self.num_outputs,
            desc=self.func.__doc__,
        )


class Dummy1to1Tool(Tool):
    def __init__(self) -> None:
        func = lambda x: x
        func.__doc__ = "Dummy 1 to 1 tool"
        super().__init__(func)

    def __str__(self) -> str:
        return "Dummy1to1Tool"

    def validate_args(self, *args) -> bool:
        return True

    @property
    def num_args(self) -> int:
        return 1

    @property
    def num_outputs(self) -> int:
        return 1

    @property
    def name(self) -> str:
        return "Dummy1to1Tool"
