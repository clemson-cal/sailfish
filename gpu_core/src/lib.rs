use std::iter::FromIterator;
use std::mem;
use std::os::raw::{c_ulong, c_void, c_int};

extern "C" {
    // Memory allocation and transfer
    pub fn gpu_malloc(size: c_ulong) -> *mut c_void;
    pub fn gpu_free(ptr: *mut c_void);
    pub fn gpu_memcpy_htod(dst: *mut c_void, src: *const c_void, size: c_ulong);
    pub fn gpu_memcpy_dtoh(dst: *mut c_void, src: *const c_void, size: c_ulong);
    pub fn gpu_memcpy_dtod(dst: *mut c_void, src: *const c_void, size: c_ulong);

    // Device control
    pub fn gpu_device_synchronize();
    pub fn gpu_get_device_count() -> c_int;
    pub fn gpu_get_device() -> c_int;
    pub fn gpu_set_device(device_id: c_int);

    // Higher level utility functions
    pub fn gpu_vec_max_f64(vec: *const f64, size: c_ulong, result: *mut f64);
}

pub struct DeviceVec<T: Copy> {
    ptr: *mut T,
    len: usize,
}

impl<T: Copy> DeviceVec<T> {
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn as_device_ptr(&self) -> *const T {
        self.ptr
    }
    pub fn as_mut_device_ptr(&mut self) -> *mut T {
        self.ptr
    }
}

impl<T: Copy> From<&[T]> for DeviceVec<T> {
    fn from(slice: &[T]) -> Self {
        let bytes = (slice.len() * mem::size_of::<T>()) as c_ulong;
        unsafe {
            let ptr = gpu_malloc(bytes);
            gpu_memcpy_htod(ptr, slice.as_ptr() as *const c_void, bytes);
            Self {
                ptr: ptr as *mut T,
                len: slice.len(),
            }
        }
    }
}

impl<T: Copy> From<&Vec<T>> for DeviceVec<T> {
    fn from(vec: &Vec<T>) -> Self {
        vec.as_slice().into()
    }
}

impl<T: Copy> From<&DeviceVec<T>> for Vec<T>
where
    T: Default,
{
    fn from(dvec: &DeviceVec<T>) -> Self {
        let mut hvec = vec![T::default(); dvec.len()];
        let bytes = (dvec.len() * mem::size_of::<T>()) as c_ulong;
        unsafe {
            gpu_memcpy_dtoh(
                hvec.as_mut_ptr() as *mut c_void,
                dvec.ptr as *const c_void,
                bytes,
            )
        };
        hvec
    }
}

impl<T: Copy> Drop for DeviceVec<T> {
    fn drop(&mut self) {
        unsafe { gpu_free(self.ptr as *mut c_void) }
    }
}

impl<T: Copy> Clone for DeviceVec<T> {
    fn clone(&self) -> Self {
        let bytes = (self.len * mem::size_of::<T>()) as c_ulong;
        unsafe {
            let ptr = gpu_malloc(bytes);
            gpu_memcpy_dtod(ptr, self.ptr as *const c_void, bytes);
            Self {
                ptr: ptr as *mut T,
                len: self.len,
            }
        }
    }
}

impl<T: Copy> FromIterator<T> for DeviceVec<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let hvec: Vec<T> = iter.into_iter().collect();
        DeviceVec::from(&hvec)
    }
}

pub trait Reduce {
    type Item: Copy;
    fn maximum(&self) -> Option<Self::Item>;
}

impl Reduce for DeviceVec<f64> {
    type Item = f64;
    fn maximum(&self) -> Option<Self::Item> {
        if self.is_empty() {
            None
        } else {
            let mut result = DeviceVec::from(&vec![0.0]);
            unsafe { gpu_vec_max_f64(self.ptr, self.len as c_ulong, result.as_mut_device_ptr()) };
            Some(Vec::from(&result)[0])
        }
    }
}

pub fn device_synchronize() {
    unsafe {
        gpu_device_synchronize()
    }
}
pub fn device_count() -> i32 {
    unsafe {
        gpu_get_device_count()
    }
}
pub fn get_device() -> i32 {
    unsafe {
        gpu_get_device()
    }
}
pub fn set_device(device_id: i32) {
    unsafe {
        gpu_set_device(device_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let hvec: Vec<_> = (0..100).collect();
        let dvec = DeviceVec::from(&hvec);
        assert_eq!(hvec, Vec::from(&dvec));
    }

    #[test]
    fn reduce() {
        for n in 0..1000 {
            let dvec: DeviceVec<_> = (0..n).map(|i| i as f64).collect();
            assert_eq!(
                dvec.maximum(),
                if n == 0 { None } else { Some((n - 1) as f64) }
            )
        }
    }

    #[test]
    fn device_get_set() {
        unsafe {
            let count = device_count();

            for id in 0..count {
                gpu_set_device(id);
                assert_eq!(gpu_get_device(), id);
            }
        }
    }
}
