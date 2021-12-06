"""
Analyzes C code intended for JIT-compilation to a kernel library.
"""


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

    The C code needs to conform to a set of conventions, which are still
    evolving but will be documented soon. This function returns a dictionary
    whose keys are the names of the public functions (or kernels) in the code,
    and the values are lists of the (positional) arguments describing the
    function signature. Each function argument is (currently) a tuple of the
    data type, the argument name, and an optional constraint which could be
    validated at runtime.
    """
    api = dict()
    for event, value in scan(code.splitlines()):
        if event == "start_symbol":
            args = []
            name = value
        elif event == "argument":
            args.append(value)
        elif event == "end_symbol":
            api[name] = args
    return api
