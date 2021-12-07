Kernel libraries
================

.. py:currentmodule:: sailfish.kernel.library

Sailfish has a module dedicated to generating CPU-GPU agnostic compute
kernels. A compute kernel is a function that operates on one or more arrays of
floating point data, and may be parallelized using OpenMP, CUDA, or ROCm.
Kernels are written in C, and the source code is provided to the kernel module
as a string. That means the source can be loaded from a file, embedded in
Python code as a string literal, or even generated programmatically. The
source code is compiled just-in-time (JIT) when the program starts, yielding a
:py:class:`Library` object. Kernels are invoked using attribute access on the
library object.

Quick example
~~~~~~~~~~~~~

Below is a minimal working example of compiling and executing a custom kernel
from an inline Python string:

.. code-block:: python

    import numpy as np
    from sailfish.kernel.library import Library

    code = """
    PUBLIC void my_kernel(
        int ni,
        int nj,
        double *data) // :: $.shape == (ni, nj)
    {
        FOR_EACH_2D(ni, nj)
        {
            data[i * nj + j] = i + j;
        }
    }
    """

    library = Library(code, mode="cpu")
    data = np.zeros([10, 20])
    library.my_kernel[data.shape](data)

In this example, a 2D numpy array of double-precision zeros was created, and
then populated by the JIT-compiled kernel. The execution mode is "cpu"
indicates sequential processing, as opposed to parallelized OpenMP or GPU
processing. Several conventions are assumed for the C code, including the
:py:obj:`PUBLIC` and :py:obj:`FOR_EACH_2D` macros, and the end-of-line
comments on the function signature. These things are explained in more detail
below.

Invoking the kernel
~~~~~~~~~~~~~~~~~~~

Let's start with how to use a kernel that's already been compiled
successfully. The library object (:py:obj:`lib` in the example above) has one
attribute per :py:obj:`PUBLIC` function defined in the kernel source. Since
kernels can in general operate on several arrays at once, and these arrays can
have distinct shapes, there's no way (in general) for the range of array
indexes to be inferred from the array arguments. For this reason, the kernel
object needs to be provided with the index space of the array to be traversed.
The code below shows the object types involved in the kernel invocation:

.. code-block:: python

    library                           # Library
    library.my_kernel                 # Kernel
    library.my_kernel[(10, 20)]       # KernelInvocation
    library.my_kernel[(10, 20)](data) # None (kernel modifies data array)

Array dimensionality
^^^^^^^^^^^^^^^^^^^^

Kernels of rank 1, 2, and 3 are supported, corresponding to 1D, 2D, or 3D
array traversals. Note that the arrays supplied as arguments to the kernel can
have different dimensionality than the rank of the index space to be
traversed; it's common for a triple-nested loop (3D index space) to operate on
a 4- or even 5-dimensional array, since each 3D index can point to an array or
matrix of data fields.

Type and bounds checking
^^^^^^^^^^^^^^^^^^^^^^^^

Kernel arguments are type-checked at runtime to match the signature of the
compute kernel. The source code is also allowed to specify additional
constraints on the shape of the array arguments, and even the values of
non-array arguments. It's highly recommended to specify the expected array
shapes in the kernel code, because this is the only protection you have
against memory corruption errors from reading or writing out-of-bounds. Type
and bounds checking does incur a small overhead, and that can become
significant if the array size is relatively small. For this reason, you might
want to disable it once your kernel and the code invoking it are stabilized.
To disable the argument validation, pass :code:`debug=False` to the
:py:obj:`Library` constructor. Just remember that when checking is disabled,
all kinds of memory corruption errors can be caused by invoking a kernel with
the wrong numer or type of arguments, or with arrays of unexpected shape.

At present, only the data types `int`, `double`, and `double*` are permitted
to be kernel arguments. More native data types can be supported as needed, but
I don't plan on enabling arbitrary data structures as kernel arguments. Just
keep your kernel signatures very simple.

Kernel source code
~~~~~~~~~~~~~~~~~~

Source conventions
^^^^^^^^^^^^^^^^^^

The sailfish kernel module doesn't use a general-purpose tool for parsing C
code, it's just based on a few regular expressions to crudely extract the
function names, signatures, and the argument constraints. For this reason, the
C code needs to follow several conventions for the parser to understand it:

- Kernel functions must start with :py:obj:`PUBLIC void`
- Arguments must go on separate lines
- The number of leading `int` arguments is used to infer the kernel rank, and
  this must be 1, 2, or 3. If the kernel needs additional `int` arguments that
  aside from those specifying the index space to be traversed, then put those
  arguments later in the signature.
- The :py:obj:`FOR_EACH_1D` macro (and 2D/3D counterparts) must be used to
  start the scope of the function body to be applied to each array element.
  This macro defines the loop variables `i, j, k` as appropriate for the
  execution strategy.
