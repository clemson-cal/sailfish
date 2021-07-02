use std::os::raw::{c_void, c_ulong};
use std::marker::PhantomData;
use std::mem;

extern "C" {
    pub fn gpu_malloc(size: c_ulong) -> *mut c_void;
    pub fn gpu_free(ptr: *mut c_void);
    pub fn gpu_memcpy_htod(dst: *mut c_void, src: *const c_void, size: c_ulong);
    pub fn gpu_memcpy_dtoh(dst: *mut c_void, src: *const c_void, size: c_ulong);
    pub fn gpu_memcpy_dtod(dst: *mut c_void, src: *const c_void, size: c_ulong);
}

pub struct DeviceVec<T: Copy> {
    ptr: *mut c_void,
    len: usize,
    phantom: PhantomData<T>
}

impl<T: Copy> DeviceVec<T> {
    pub fn len(&self) -> usize {
        self.len
    }
}

impl<T: Copy> From<&[T]> for DeviceVec<T> {
    fn from(slice: &[T]) -> Self {
        let bytes = (slice.len() * mem::size_of::<T>()) as c_ulong;
        unsafe {
            let ptr = gpu_malloc(bytes);
            gpu_memcpy_htod(ptr, slice.as_ptr() as *const c_void, bytes);
            Self {
                ptr,
                len: slice.len(),
                phantom: PhantomData,
            }            
        }
    }
}

impl<T: Copy> From<&Vec<T>> for DeviceVec<T> {
    fn from(vec: &Vec<T>) -> Self {
        vec.as_slice().into()
    }
}

impl<T: Copy> From<&DeviceVec<T>> for Vec<T> where T: Default {
    fn from(dvec: &DeviceVec<T>) -> Self {
        let mut hvec = vec![T::default(); dvec.len()];
        let bytes = (dvec.len() * mem::size_of::<T>()) as c_ulong;
        unsafe { gpu_memcpy_dtoh(hvec.as_mut_ptr() as *mut c_void, dvec.ptr, bytes) };
        hvec
    }
}

impl<T: Copy> Drop for DeviceVec<T> {
    fn drop(&mut self) {
        unsafe {
            gpu_free(self.ptr)
        }
    }
}

impl<T: Copy> Clone for DeviceVec<T> {
    fn clone(&self) -> Self {
        let bytes = (self.len * mem::size_of::<T>()) as c_ulong;
        unsafe {
            let ptr = gpu_malloc(bytes);
            gpu_memcpy_dtod(ptr, self.ptr, bytes);
            Self {
                ptr,
                len: self.len,
                phantom: self.phantom,
            }            
        }
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
}
