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
    unsafe fn alloc<T: Copy>(&self, len: usize) -> DeviceBuffer<T> {
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
    pub fn device(&self) -> Device {
        Device(self.device_id)
    }
    pub fn copy_to(&self, device: Device) -> Self {
        unsafe {
            let buffer = device.alloc(self.len());
            gpu_memcpy_peer(
                buffer.ptr as *mut c_void,
                device.0,
                self.ptr as *const c_void,
                self.device_id,
                (self.len() * size_of::<T>()) as c_ulong,
            );
            buffer
        }
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
}
