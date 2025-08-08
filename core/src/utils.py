from dataclasses import dataclass

TYPES_MAP: dict[str, type] = {
    "<class 'int'>": int,
    "<class 'str'>": str,
    "<class 'float'>": float,
    "<class 'bool'>": bool,
    "<class 'list'>": list,
    "<class 'tuple'>": tuple,
    "<class 'dict'>": dict,
    "<class 'set'>": set,
    "int": int,
    "str": str,
    "float": float,
    "bool": bool,
    "list": list,
    "tuple": tuple,
    "dict": dict,
    "set": set,
}


@dataclass
class Arg:
    name: str
    type: type

    @classmethod
    def from_str(cls, arg_str: str) -> "Arg":
        name, type = arg_str.split(":")
        return cls(name=name, type=TYPES_MAP[type])

    @classmethod
    def from_name_and_type(cls, arg_name: str, arg_type: "type") -> "Arg":
        return cls(name=arg_name, type=arg_type)


def func_input_args(func) -> list[Arg]:
    assert func.__annotations__, "Function must have annotations"
    return [
        Arg.from_str(f"{arg_name}:{arg_type_repr}")
        for arg_name, arg_type_repr in func.__annotations__.items()
        if arg_name != "return"
    ]


def func_output_args(func) -> list[Arg]:
    assert func.__annotations__, "Function must have annotations"
    assert "return" in func.__annotations__, "Function must have a return annotation"

    return_annotations: type = func.__annotations__["return"]
    # Check if the return type is a tuple
    if isinstance(return_annotations, tuple):
        return_annotations_repr: str = str(return_annotations)
        return_annotations_repr = return_annotations_repr[len("tuple") + 1 : -1]
        return_annotations_list: list[type] = [
            TYPES_MAP[x.strip()] for x in return_annotations_repr.split(",")
        ]
    else:
        return_annotations_list: list[type] = [return_annotations]

    return [
        Arg.from_name_and_type(f"output:{i}", type_repr)
        for i, type_repr in enumerate(return_annotations_list)
    ]
