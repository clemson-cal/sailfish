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

code = """
PRIVATE double twice(double x)
{
    return x * 2.0;
}

PUBLIC void my_kernel(
    int ni,
    double *data) // :: $.shape == (ni,)
{
    FOR_EACH_1D(ni)
    {
        data[i] = twice(i);
    }
}
"""

library = Library(code, mode="cpu", name="my_module")
data = np.zeros(10)
library.my_kernel(len(data), data)

for i in data:
    print(i)
