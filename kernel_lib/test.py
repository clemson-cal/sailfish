import numpy as np
from kernel_lib.library import Library

# Several implicit things:
#
# - If a C kernel has an n leading integer arguments, then the kernel launch
#   is n-dimensional.
#
# - Functions that are part of the API need to start with PUBLIC. Other
#   functions start with PRIVATE.
#
# - The signatures of PUBLIC functions need to go on separate lines.
#
# - For loops are replaced with a macro,
#
# - Function arguments can only be int, double, and double*.

code = open("example.c").read()
library = Library(code, mode="cpu", name="my_module")
data = np.zeros(10)

library.my_1d_kernel([len(data)], data)

for i in data:
    print(i)
