"""
Illustrates and profiles concurrent kernel launches (GPU only).

This program is also useful in determining how big an array needs to be to
"saturate" (i.e. max out the occupancy of) the GPU. When 
"""

from cupy import zeros
from cupy.cuda import Stream
from numpy.typing import NDArray
from new_kernels import kernel, perf_time_sequence


code = R"""
PUBLIC void compute(int ni, int nj, int count, double *x)
{
    int sj = 1;
    int si = nj;

    FOR_EACH_2D(ni, ni) {
        double x1 = 0.0;

        for (int m = 0; m < count; ++m)
        {
            x1 += sin((double)m) * cos((double)m);
        }
        x[i * si + j * sj] = x1;
    }
}
"""


@kernel(code, rank=2)
def compute(count: int, x: NDArray[float]):
    return x.shape


if __name__ == "__main__":
    iterations_per_element = 10000
    array_size = (200, 200)
    num_arrays = 10
    num_samples = 20

    arrays = list(zeros(array_size) for _ in range(num_arrays))
    streams = list(Stream() for _ in range(num_arrays))

    for _, time_taken in zip(range(num_samples), perf_time_sequence(mode="gpu")):

        for stream, array in zip(streams, arrays):
            compute(iterations_per_element, array, exec_mode="gpu", stream=stream)

        work = iterations_per_element * array_size[0] * array_size[1] * num_arrays
        print(f"time={work/time_taken/1e9:.2f} Gzps")
