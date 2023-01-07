"""
A script to invoke the sailfish command line entry point.

This script is just here for backwards compatibility. Its official location is
now bin/sailfish.
"""

from sys import path
from pathlib import Path

path.append(str(Path(__file__).parent.parent))

from sailfish.driver import main

main()
