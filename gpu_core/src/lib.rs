use cfg_if::cfg_if;

pub mod buffer;
pub mod device;
pub use buffer::Buffer;
pub use device::Device;

cfg_if! {
    if #[cfg(feature = "gpu")] {
        use std::os::raw::{c_int, c_ulong, c_void, c_char};
        pub mod device_buffer;
        pub use device_buffer::{DeviceBuffer, Reduce};
    }
}

#[cfg(feature = "gpu")]
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
        src: *const c_void,
        dst_start_i: c_ulong,
        dst_start_j: c_ulong,
        dst_start_k: c_ulong,
        dst_shape_i: c_ulong,
        dst_shape_j: c_ulong,
        dst_shape_k: c_ulong,
        src_start_i: c_ulong,
        src_start_j: c_ulong,
        src_start_k: c_ulong,
        src_shape_i: c_ulong,
        src_shape_j: c_ulong,
        src_shape_k: c_ulong,
        count_i: c_ulong,
        count_j: c_ulong,
        count_k: c_ulong,
        bytes_per_elem: c_ulong,
    );

    // Device control
    pub fn gpu_device_synchronize();
    pub fn gpu_get_device_count() -> c_int;
    pub fn gpu_get_device() -> c_int;
    pub fn gpu_set_device(device_id: c_int);
    pub fn gpu_get_last_error() -> *const c_char;

    // Higher level utility functions
    pub fn gpu_vec_max_f64(vec: *const f64, size: c_ulong, result: *mut f64);
}

/// Executes the given closure on a GPU device if `device` is `Some`.
/// Otherwise, the closure is just executed.
pub fn scope<T, F: FnMut() -> T>(device: Option<Device>, mut f: F) -> T {
    if let Some(device) = device {
        cfg_if! {
            if #[cfg(feature = "gpu")] {
                on_device(device.0, f)
            } else {
                panic!("gpu feature not enabled, requested device {:?}", device)
            }
        }
    } else {
        f()
    }
}

/// Returns an iterator over all the devices on this system.
pub fn all_devices() -> impl Iterator<Item = Device> + Clone {
    cfg_if! {
        if #[cfg(feature = "gpu")] {
            (0..unsafe { gpu_get_device_count() }).map(Device)
        } else {
            (0..0).map(Device)
        }
    }
}

#[cfg(feature = "gpu")]
fn on_device<T, F: FnMut() -> T>(device: i32, mut f: F) -> T {
    use std::ffi::CStr;
    let (result, error_str) = unsafe {
        let orig = gpu_get_device();
        gpu_set_device(device);
        let result = f();
        let error_str = gpu_get_last_error();
        let error_str = if error_str != std::ptr::null() {
            Some(CStr::from_ptr(error_str).to_str().unwrap().to_owned())
        } else {
            None
        };
        gpu_set_device(orig);
        (result, error_str)
    };
    if let Some(error_str) = error_str {
        panic!("{}", error_str)
    } else {
        result
    }
}
