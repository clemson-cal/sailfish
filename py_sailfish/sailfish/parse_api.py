def scan(lines):
    import re

    function_name = re.compile(r"\s*PUBLIC\s+void\s+(?P<symbol>\w+)")
    argument_name = re.compile(
        r"\s*(?P<dtype>\w+\s*\**)\s*(?P<argname>\w+)\s*[,\)]\s*(?:///)?\s*(?P<constraint>.*)"
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
                dtype, argname, constraint = match.groups()
                yield "argument", (dtype.replace(" ", ""), argname, constraint)
            else:
                symbol = None
                yield "end_symbol", None


def parse_api(filename):
    api = dict()
    with open(filename, "r") as f:
        for event, value in scan(f):
            if event == "start_symbol":
                args = []
                name = value
            elif event == "argument":
                args.append(value)
            elif event == "end_symbol":
                api[name] = args
    return api


if __name__ == "__main__":
    import argparse, pprint

    args = argparse.ArgumentParser()
    args.add_argument("filename", type=str)
    api = parse_api(args.parse_args().filename)
    pprint.pprint(api)
