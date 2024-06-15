from typing import Union

def str_int_arg(arg) -> Union[str, int]:
    try:
        return int(arg)
    except ValueError:
        return arg