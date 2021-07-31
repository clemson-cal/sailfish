use std::os::raw::{c_int, c_ulong, c_void};

#[cfg(feature = "gpu")]
pub mod buffer;
pub mod device;

#[cfg(feature = "gpu")]
pub use buffer::DeviceBuffer;
#[cfg(feature = "gpu")]
pub use buffer::Reduce;
pub use device::Device;

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

#[cfg(feature = "gpu")]
fn on_device<T, F: Fn() -> T>(device: i32, f: F) -> T {
    let orig = unsafe { gpu_get_device() };
    unsafe { gpu_set_device(device) };
    let result = f();
    unsafe { gpu_set_device(orig) };
    result
}

#[cfg(feature = "gpu")]
pub fn all_devices() -> impl Iterator<Item = Device> {
    (0..unsafe { gpu_get_device_count() }).map(Device)
}
