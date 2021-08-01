//! Exports a `Device` struct for interacting with the GPU device.
//!
//! The `Device` struct provides an execution context to run closures on a
//! particular device, including memory allocation and kernel launches. A stub
//! `Device` struct is provided to help with feature compatibility: creation
//! returns `None`, synchronization and scoping are no-ops, and buffer
//! creation methods are removed from the API (`gpu_core::Buffer` does not
//! provide a stub implementation).

#[cfg(feature = "gpu")]
use crate::*;

#[cfg(feature = "gpu")]
use buffer::DeviceBuffer;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Device(pub(crate) i32);

#[cfg(feature = "gpu")]
impl Device {
    /// Returns a new `Device` handle from an integer ID. Returns `Non` if a
    /// device with the given ID does not exist.
    pub fn with_id(id: i32) -> Option<Self> {
        if id < unsafe { gpu_get_device_count() } as i32 {
            Some(Self(id as i32))
        } else {
            None
        }
    }

    /// Returns a new buffer allocated on this device by copying data from the
    /// given slice.
    pub fn buffer_from<T: Copy>(&self, slice: &[T]) -> DeviceBuffer<T> {
        on_device(self.0, || {
            let bytes = (slice.len() * std::mem::size_of::<T>()) as c_ulong;
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

    /// Returns an unitialized buffer of the given size on this device.
    pub unsafe fn uninit_buffer<T: Copy>(&self, len: usize) -> DeviceBuffer<T> {
        on_device(self.0, || {
            let bytes = (len * std::mem::size_of::<T>()) as c_ulong;
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

    /// Executes the given closure on this device. This function restores the
    /// current device on this OS thread before it returns.
    pub fn scope<T, F: Fn(&Self) -> T>(&self, f: F) -> T {
        on_device(self.0, || f(self))
    }

    /// Calls `cudaDeviceSynchronize` or `hipDeviceSynchronize`.
    pub fn synchronize(&self) {
        on_device(self.0, || unsafe { gpu_device_synchronize() })
    }

    pub fn last_error(&self) -> Option<String> {
        use std::ffi::CStr;

        on_device(self.0, || {
            let error_str = unsafe { gpu_get_last_error() };
            if error_str == std::ptr::null() {
                None
            } else {
                Some(
                    unsafe { CStr::from_ptr(error_str) }
                        .to_str()
                        .unwrap()
                        .to_owned(),
                )
            }
        })
    }
}

impl Default for Device {
    fn default() -> Self {
        Self::with_id(0).unwrap()
    }
}

#[cfg(not(feature = "gpu"))]
impl Device {
    pub fn with_id(_id: i32) -> Option<Self> {
        None
    }
    pub fn scope<T, F: Fn(&Self) -> T>(&self, f: F) -> T {
        f(self)
    }
    pub fn synchronize(&self) {}
    pub fn last_error(&self) -> Option<String> {
        None
    }
}
