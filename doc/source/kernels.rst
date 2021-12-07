Kernel libraries
================

.. py:currentmodule:: sailfish.kernel.library

Sailfish has a module dedicated to generating CPU-GPU agnostic compute
kernels. A compute kernel is a function that operates on one or more arrays of
floating point data, and may be parallelized using OpenMP, CUDA, or ROCm.
Kernels are written in C, and the source code is provided to the `kernel`
module as a string. That means the source can be loaded from a file, embedded
in Python code as a string literal, or even generated programmatically. The
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
then populated by the JIT-compiled kernel. The execution mode is "cpu", which
indicates sequential processing, as opposed to parallelized OpenMP or GPU
processing. Several conventions are assumed for the C code, including the
:py:obj:`PUBLIC` and :py:obj:`FOR_EACH_2D` macros, and the end-of-line
comments on the function signature. These things are explained in more detail
below.

Invoking the kernel
~~~~~~~~~~~~~~~~~~~

Let's start with how to use a kernel that's already been compiled
successfully. The library object (:py:obj:`library` in the example above) has
one attribute per :py:obj:`PUBLIC` function defined in the kernel source.
Since kernels can in general operate on several arrays at once, and these
arrays can have distinct shapes, there's no way (in general) for the range of
array indexes to be inferred from the array arguments. For this reason, the
kernel object needs to be provided with a `traversal` index space. For the
simple example above, the traversal index space is the same as the array
argument. However this is not always the case, for example if the array has
guard zones, or if there are many fields per spatial array index. The code
below shows the object types involved in the kernel invocation:

.. code-block:: python

    library                           # Library
    library.my_kernel                 # Kernel
    library.my_kernel[(10, 20)]       # KernelInvocation
    library.my_kernel[(10, 20)](data) # None (kernel modifies data array)

Kernel rank
^^^^^^^^^^^

Kernels of rank 1, 2, and 3 are supported, corresponding to 1D, 2D, or 3D
array traversals. The arrays supplied as arguments to the kernel can have
different dimensionality than the rank of the index space to be traversed. For
example, a triple-nested loop (rank-3 index space) to operate on a 4D array,
where multiple fields are stored at each `(i,j,k)` index.

Type and bounds checking
^^^^^^^^^^^^^^^^^^^^^^^^

Kernel arguments are type-checked at runtime to match the signature of the
compute kernel. The kernel source code is also allowed to specify additional
constraints on the shape of the array arguments, and even the values of
non-array arguments. It's highly recommended to at least specify the expected
array shapes in the kernel code, because this is the only protection you have
against memory corruption errors due to passing arrays with unexpected shape
to the kernel. Type and bounds checking does incur a small overhead, and that
can become significant if the array size is relatively small. For this reason,
you might want to disable it once your kernel and the Python code invoking it
are stabilized. To disable the argument validation, pass :code:`debug=False`
to the :py:obj:`Library` constructor. Just remember that when checking is
disabled, all kinds of memory corruption errors can be caused by invoking a
kernel with the wrong numer or type of arguments, or with arrays of unexpected
shape.

At present, only the data types `int`, `double`, and `double*` are permitted
to be kernel arguments. More native data types can be supported as needed, but
I don't plan on enabling arbitrary data structures as kernel arguments. Just
keep your kernel signatures very simple.

Building a kernel
~~~~~~~~~~~~~~~~~

Kernel compilation takes place when the :py:obj:`Library` object is
instantiated. The constructor is given a string of source code, and a
compilation mode, which is a string with one of the following values:

- :code:`cpu` kernel body is embedded in a sequential for-loop; compiled with `CFFI`
- :code:`omp` kernel body is embedded in an OpenMP-annotated for-loop; compiled with `CFFI`
- :code:`gpu` kernel body is executed once per GPU thread; compiled with `cupy`

These execution modes are facilitated by the 1D, 2D, and 3D versions of the
:py:obj:`FOR_EACH` preprocessor directives. Those directives take on different
values depending on the execution mode (see the :file:`library.py` source-code
to see how this works).

**Implementation note**: When compiling kernels for CPU execution, the `CFFI`
module leaves behind files on the disk, including a generated C file and the
build product, which is a shared library (`.so`) file. The shared library is
loaded using :py:obj:`ctypes.CDLL`, but after it's loaded it would be fine to
remove the build products from the file system. However, the `kernel` module
caches the shared library files to reduce program startup time from the JIT
compilation. The cached libraries are kept in the
`sailfish/kernel/__pycache__` directory, and identified by the SHA hash of the
source code itself, combined with any preprocessor directives. For this
reason, your cache directory will accumulate many stale build products if you
are modifying the kernel sources frequently. It's always safe to delete a
`__pycache__` directory. No caching is done for GPU builds.

Kernel source code
~~~~~~~~~~~~~~~~~~

Source conventions
^^^^^^^^^^^^^^^^^^

The sailfish `kernel` module doesn't use a general-purpose tool for parsing C
code, it's just based on a few regular expressions to crudely extract the
function names, signatures, and the argument constraints. For this reason, the
C code needs to follow several conventions for the parser to understand it:

- Kernel functions must start with :py:obj:`PUBLIC void`
- Helper functions must start with :py:obj:`PRIVATE`; they are not accessible
  to Python code
- Arguments must go on separate lines
- The number of leading `int` arguments is used to infer the kernel rank, and
  this must be 1, 2, or 3. If the kernel needs additional `int` arguments
  aside from those specifying the index space to be traversed, then put those
  arguments later in the signature.
- The :py:obj:`FOR_EACH_1D` macro (and 2D/3D counterparts) must be used to
  start the scope of the function body to be applied to each array element.
  This macro defines the loop variables `i, j, k` as appropriate for the
  execution strategy.

Argument constraints
^^^^^^^^^^^^^^^^^^^^

Argument contraints are Python expressions, embedded in the C code, and
associated with a kernel function argument. The Python expression must go on
the same line as the kernel argument (another reason the parser requires one
argument per line in the function signature), on a new-line comment, and
double-colon, like this:

.. code-block:: C

    PUBLIC void constrained_kernel(
        int ni,
        int nj,
        double *coordinates, // :: $.shape == (ni, nj, 2)
        double *fields,      // :: $.shape == (ni, nj, num_fields)
        double time,         // :: $ > 1.0
        int num_fields)      // :: $ > 0
    {
        // kernel body here
    }

This example is a rank-2 kernel, and the shape of the traversed index space is
:py:obj:`(ni, nj)`. Two arrays are passed in: an array of coordinates to be
read from, and an array of fields to be written to. The shapes of the two
arrays are constrained, and an exception would be raised if the shapes did not
match (unless debug mode was disabled). Constraint expressions are evaluated
in a Python scope that contains the values of the other arguments, so the
constraints can be relative to other arguments provided.

Keep in mind that argument constraints are optional, but that including them
on the array arguments is the only way to ensure any level of memory safety.
Including them is also good because it documents your C code with the array
shapes expected by the kernel.
