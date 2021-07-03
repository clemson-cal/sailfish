use crate::index_space::IndexSpace;

#[derive(Clone, Debug)]
pub struct Patch {
    /// The region of index space covered by this patch.
    space: IndexSpace,

    /// The number of fields stored at each zone.
    num_fields: usize,

    /// The backing array of data for this patch.
    data: Vec<f64>,
}

#[cfg(feature = "cuda")]
#[derive(Clone, Debug)]
pub struct DevicePatch {
    /// The region of index space covered by this patch.
    space: IndexSpace,

    /// The number of fields stored at each zone.
    num_fields: usize,

    /// The device allocation for this patch.
    data: gpu_mem::DeviceVec<f64>,
}
