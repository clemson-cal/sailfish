"""
Analyzes C code intended for JIT-compilation to a kernel library.
"""

from typing import NamedTuple, List


class Argument(NamedTuple):
    dtype: str
    name: str
    constraint: str


class Symbol(NamedTuple):
    name: str
    args: List[Argument]

    @property
    def rank(self):
        r = 0
        for arg in self.args:
            if arg.dtype == "int":
                r += 1
            else:
                break
        return r


def scan(lines):
    """
    Generator to emit events encountered parsing kernel library code.
    """
    import re

    function_name = re.compile(r"\s*PUBLIC\s+void\s+(?P<symbol>\w+)")
    argument_name = re.compile(
        r"\s*(?P<dtype>\w+\s*\**)\s*(?P<argname>\w+)\s*[,\)]\s*(?://)?\s*(?P<comment>.*)"
    )
    symbol = None
    for line in lines:
        if not symbol:
            match = function_name.match(line)
            if match is not None:
                symbol = match.group("symbol")
                yield "start_symbol", symbol
        else:
            match = argument_name.match(line)
            if match is not None:
                dtype, argname, comment = match.groups()
                dtype = dtype.replace(" ", "")
                constraint = comment.partition("::")[2]
                yield "argument", (dtype, argname, constraint)
            else:
                symbol = None
                yield "end_symbol", None


def parse_api(code):
    """
    Parse a C-like source file to extract a public API.

    The C code needs to conform to a set of conventions, which still need to be
    fully documented. This function returns a dictionary whose keys are the
    names of the public functions (or kernels) in the code, and the values are
    lists of the (positional) arguments describing the function signature. Each
    function argument is a tuple of the data type, the argument name, and an
    optional constraint which could be validated at runtime.
    """
    api = dict()
    for event, value in scan(code.splitlines()):
        if event == "start_symbol":
            args = []
            name = value
        elif event == "argument":
            args.append(Argument(*value))
        elif event == "end_symbol":
            api[name] = Symbol(name=name, args=args)

    for symbol in api.values():
        if not 1 <= symbol.rank <= 3:
            raise ValueError(
                f"kernel {symbol} has rank {symbol.rank}, must be 1, 2, or 3"
            )
    return api


def main():
    import argparse, pprint

    args = argparse.ArgumentParser()
    args.add_argument("filename", type=str)
    with open(args.parse_args().filename) as f:
        api = parse_api(f.read())

    for symbol in api.values():
        print(
            f"{symbol.name} (rank {symbol.rank}) ({', '.join(f'{a.name}: {a.dtype}' for a in symbol.args)})"
        )


if __name__ == "__main__":
    main()
