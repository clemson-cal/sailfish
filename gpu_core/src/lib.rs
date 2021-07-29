use std::mem::size_of;
use std::os::raw::{c_int, c_ulong, c_void};

extern "C" {
    // Memory allocation and transfer
    pub fn gpu_malloc(size: c_ulong) -> *mut c_void;
    pub fn gpu_free(ptr: *mut c_void);
    pub fn gpu_memcpy_htod(dst: *mut c_void, src: *const c_void, size: c_ulong);
    pub fn gpu_memcpy_dtoh(dst: *mut c_void, src: *const c_void, size: c_ulong);
    pub fn gpu_memcpy_dtod(dst: *mut c_void, src: *const c_void, size: c_ulong);
    pub fn gpu_memcpy_peer(
        dst: *mut c_void,
        dst_device: c_int,
        src: *const c_void,
        src_device: c_int,
        size: c_ulong,
    );
    pub fn gpu_memcpy_3d(
        dst: *mut c_void,
        start_dst: *const c_ulong,
        shape_dst: *const c_ulong,
        src: *const c_void,
        start_src: *const c_ulong,
        shape_src: *const c_ulong,
        count: *const c_ulong,
        bytes: c_ulong,
    );

    // Device control
    pub fn gpu_device_synchronize();
    pub fn gpu_get_device_count() -> c_int;
    pub fn gpu_get_device() -> c_int;
    pub fn gpu_set_device(device_id: c_int);

    // Higher level utility functions
    pub fn gpu_vec_max_f64(vec: *const f64, size: c_ulong, result: *mut f64);
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Device(i32);

impl Device {
    pub fn with_id(id: i32) -> Option<Self> {
        if id < unsafe { gpu_get_device_count() } as i32 {
            Some(Self(id as i32))
        } else {
            None
        }
    }
    pub fn buffer_from<T: Copy>(&self, slice: &[T]) -> DeviceBuffer<T> {
        on_device(self.0, || {
            let bytes = (slice.len() * size_of::<T>()) as c_ulong;
            unsafe {
                let ptr = gpu_malloc(bytes);
                gpu_memcpy_htod(ptr, slice.as_ptr() as *const c_void, bytes);
                DeviceBuffer {
                    ptr: ptr as *mut T,
                    len: slice.len(),
                    device_id: self.0,
                }
            }
        })
    }
    pub unsafe fn uninit_buffer<T: Copy>(&self, len: usize) -> DeviceBuffer<T> {
        on_device(self.0, || {
            let bytes = (len * size_of::<T>()) as c_ulong;
            unsafe {
                let ptr = gpu_malloc(bytes);
                DeviceBuffer {
                    ptr: ptr as *mut T,
                    len,
                    device_id: self.0,
                }
            }
        })
    }
    pub fn scope<T, F: Fn(&Self) -> T>(&self, f: F) -> T {
        on_device(self.0, || f(self))
    }
    pub fn synchronize(&self) {
        on_device(self.0, || unsafe { gpu_device_synchronize() })
    }
}

impl Default for Device {
    fn default() -> Self {
        Self::with_id(0).unwrap()
    }
}

pub fn all_devices() -> impl Iterator<Item = Device> {
    (0..unsafe { gpu_get_device_count() }).map(Device)
}

pub struct DeviceBuffer<T: Copy> {
    ptr: *mut T,
    len: usize,
    device_id: i32,
}

impl<T: Copy> DeviceBuffer<T> {
    /// Returns whether there are any elements in this buffer.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the number of elements in this buffer
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns a typed, immutable pointer to the device allocation.
    pub fn as_device_ptr(&self) -> *const T {
        self.ptr
    }

    /// Returns a typed, mutable pointer to the device allocation.
    pub fn as_mut_device_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Returns a handle for the compute device hosting this buffer.
    pub fn device(&self) -> Device {
        Device(self.device_id)
    }

    /// Convenience method to copy memory to a `Vec`.
    pub fn to_vec(&self) -> Vec<T>
    where
        T: Default,
    {
        Vec::from(self)
    }

    /// Copy this buffer to a buffer on another device.
    pub fn copy_to(&self, device: Device) -> Self {
        on_device(self.device_id, || unsafe {
            let buffer = device.uninit_buffer(self.len());
            gpu_memcpy_peer(
                buffer.ptr as *mut c_void,
                device.0,
                self.ptr as *const c_void,
                self.device_id,
                (self.len() * size_of::<T>()) as c_ulong,
            );
            buffer
        })
    }

    /// Insert the contents of a contiguous buffer into a subset of this one.
    /// This buffer must have `shape` indexes on each axis, `elems` data
    /// elements per index, and `start + count` indexes must in-bounds on each
    /// axis. The other buffer must have `count` indexes on each axis, and
    /// `elems` data elements per index. If the other buffer resides on
    /// another device, its contents will first be copied to the destination
    /// buffer's device.
    pub fn insert_3d(
        &mut self,
        start: [usize; 3],
        shape: [usize; 3],
        count: [usize; 3],
        elems: usize,
        src_array: &Self,
    ) {
        if self.device() != src_array.device() {
            self.insert_3d(
                start,
                shape,
                count,
                elems,
                &src_array.copy_to(self.device()),
            )
        } else {
            self.memcpy_3d(start, shape, src_array, [0, 0, 0], count, count, elems)
        }
    }

    /// Extract the contents of a buffer into a new buffer. This buffer must
    /// have `shape` indexes on each axis, `elems` data elements per index,
    /// and `start + count` indexes must in-bounds on each axis. The other
    /// buffer must have `count` indexes on each axis, and `elems` data
    /// elements per index. The new buffer will reside on the same device as
    /// this one.
    pub fn extract_3d(
        &self,
        start: [usize; 3],
        shape: [usize; 3],
        count: [usize; 3],
        elems: usize,
    ) -> Self {
        let mut dst_array = unsafe {
            self.device()
                .uninit_buffer(count[0] * count[1] * count[2] * elems)
        };
        dst_array.memcpy_3d([0, 0, 0], count, self, start, shape, count, elems);
        dst_array
    }

    /// Copy from a subset of a source buffer into a subset of this buffer.
    /// Both buffers must reside on the same device.
    #[allow(clippy::too_many_arguments)]
    pub fn memcpy_3d(
        &mut self,
        dst_start: [usize; 3],
        dst_shape: [usize; 3],
        src_array: &Self,
        src_start: [usize; 3],
        src_shape: [usize; 3],
        count: [usize; 3],
        elems: usize,
    ) {
        assert_eq!(self.device(), src_array.device());
        assert_eq!(
            src_shape[0] * src_shape[1] * src_shape[2] * elems,
            src_array.len()
        );
        assert_eq!(
            dst_shape[0] * dst_shape[1] * dst_shape[2] * elems,
            self.len()
        );
        assert!(dst_start[0] + count[0] <= dst_shape[0]);
        assert!(dst_start[1] + count[1] <= dst_shape[1]);
        assert!(dst_start[2] + count[2] <= dst_shape[2]);
        assert!(src_start[0] + count[0] <= src_shape[0]);
        assert!(src_start[1] + count[1] <= src_shape[1]);
        assert!(src_start[2] + count[2] <= src_shape[2]);

        let ulong_buf = |a: [usize; 3]| {
            self.device()
                .buffer_from(&[a[0] as c_ulong, a[1] as c_ulong, a[2] as c_ulong])
        };
        let dst_start = ulong_buf(dst_start);
        let dst_shape = ulong_buf(dst_shape);
        let src_start = ulong_buf(src_start);
        let src_shape = ulong_buf(src_shape);
        let count = ulong_buf(count);

        on_device(self.device_id, || unsafe {
            gpu_memcpy_3d(
                self.ptr as *mut c_void,
                dst_start.as_device_ptr(),
                dst_shape.as_device_ptr(),
                src_array.ptr as *const c_void,
                src_start.as_device_ptr(),
                src_shape.as_device_ptr(),
                count.as_device_ptr(),
                (elems * size_of::<T>()) as c_ulong,
            )
        })
    }
}

impl<T: Copy> From<&DeviceBuffer<T>> for Vec<T>
where
    T: Default,
{
    fn from(dvec: &DeviceBuffer<T>) -> Self {
        on_device(dvec.device_id, || {
            let mut hvec = vec![T::default(); dvec.len()];
            let bytes = (dvec.len() * size_of::<T>()) as c_ulong;
            unsafe {
                gpu_memcpy_dtoh(
                    hvec.as_mut_ptr() as *mut c_void,
                    dvec.ptr as *const c_void,
                    bytes,
                )
            };
            hvec
        })
    }
}

impl<T: Copy> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        on_device(self.device_id, || unsafe {
            gpu_free(self.ptr as *mut c_void)
        })
    }
}

impl<T: Copy> Clone for DeviceBuffer<T> {
    fn clone(&self) -> Self {
        on_device(self.device_id, || {
            let bytes = (self.len * size_of::<T>()) as c_ulong;
            unsafe {
                let ptr = gpu_malloc(bytes);
                gpu_memcpy_dtod(ptr, self.ptr as *const c_void, bytes);
                Self {
                    ptr: ptr as *mut T,
                    len: self.len,
                    device_id: self.device_id,
                }
            }
        })
    }
}

/// This impl is required because of the *mut T data member. The symantics of
/// the struct member functions, and the thread safety of the CUDA and HIP
/// API's, guarantee it's safe to transfer ownership of a device buffer across
/// thread boundaries.
unsafe impl<T: Copy> Send for DeviceBuffer<T> {}
unsafe impl<T: Copy> Sync for DeviceBuffer<T> {}

pub trait Reduce {
    type Item: Copy;
    fn maximum(&self) -> Option<Self::Item>;
}

impl Reduce for DeviceBuffer<f64> {
    type Item = f64;
    fn maximum(&self) -> Option<Self::Item> {
        if self.is_empty() {
            None
        } else {
            self.device().scope(|device| {
                let mut result = device.buffer_from(&[0.0]);
                unsafe {
                    gpu_vec_max_f64(self.ptr, self.len as c_ulong, result.as_mut_device_ptr())
                };
                Some(Vec::from(&result)[0])
            })
        }
    }
}

fn on_device<T, F: Fn() -> T>(device: i32, f: F) -> T {
    let orig = unsafe { gpu_get_device() };
    unsafe { gpu_set_device(device) };
    let result = f();
    unsafe { gpu_set_device(orig) };
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let gpu = Device::default();
        let hvec: Vec<_> = (0..100).collect();
        let dvec = gpu.buffer_from(&hvec);
        assert_eq!(hvec, Vec::from(&dvec));
    }

    #[test]
    fn peer_to_peer() {
        for src in all_devices() {
            for dst in all_devices() {
                let hvec: Vec<_> = (0..100).collect();
                let dvec1 = src.buffer_from(&hvec);
                let dvec2 = dvec1.copy_to(dst);
                assert_eq!(Vec::from(&dvec1), Vec::from(&dvec2));
            }
        }
    }

    #[test]
    fn reduce() {
        for device in all_devices() {
            for n in 0..1000 {
                let hvec: Vec<_> = (0..n).map(|i| i as f64).collect();
                let dvec = device.buffer_from(&hvec);
                assert_eq!(
                    dvec.maximum(),
                    if n == 0 { None } else { Some((n - 1) as f64) }
                )
            }
        }
    }

    #[test]
    fn multi_dimensional_copy_trivial() {
        let ni = 5;
        let nj = 9;
        let nk = 8;
        let nq = 3;
        let si = nq * nk * nj;
        let sj = nq * nk;
        let sk = nq;
        let mut hvec = vec![0.0; ni * nj * nk * nq];

        let f = |i, j, k, q| (i + j + k + q) as f64;
        for i in 0..ni {
            for j in 0..nj {
                for k in 0..nk {
                    for q in 0..nq {
                        hvec[i * si + j + sj + k * sk + q] = f(i, j, k, q);
                    }
                }
            }
        }
        let dvec = Device::default().buffer_from(&hvec);
        let gvec = dvec
            .extract_3d([0, 0, 0], [ni, nj, nk], [ni, nj, nk], nq)
            .to_vec();
        assert_eq!(hvec, gvec);
    }

    #[test]
    fn multi_dimensional_copy_harder() {
        let ni = 5;
        let nj = 9;
        let nk = 8;
        let nq = 3;
        let si = nq * nk * nj;
        let sj = nq * nk;
        let sk = nq;
        let mut hvec = vec![0.0; ni * nj * nk * nq];

        let f = |i, j, k, q| (i + j + k + q) as f64;
        for i in 0..ni {
            for j in 0..nj {
                for k in 0..nk {
                    for q in 0..nq {
                        hvec[i * si + j + sj + k * sk + q] = f(i, j, k, q);
                    }
                }
            }
        }
        let dvec = Device::default().buffer_from(&hvec);
        let mi = 1;
        let mj = 2;
        let mk = 3;
        let mq = nq;
        let hsub = dvec
            .extract_3d([2, 3, 4], [ni, nj, nk], [mi, mj, mk], nq)
            .to_vec();
        let ti = mq * mk * mj;
        let tj = mq * mk;
        let tk = mq;

        for i in 2..mi {
            for j in 3..mj {
                for k in 4..mk {
                    for q in 0..mq {
                        assert_eq!(
                            hsub[(i - 2) * ti + (j - 3) * tj + (k - 4) * tk + q],
                            f(i, j, k, q)
                        );
                    }
                }
            }
        }
    }
}
