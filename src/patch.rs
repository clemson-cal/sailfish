#[cfg(feature = "gpu")]
use gpu_core::Device;
use gridiron::index_space::IndexSpace;
use gridiron::rect_map::Rectangle;
use serde::ser::{Serialize, Serializer};
use std::mem::size_of;
use Buffer::*;

#[derive(Clone)]
enum Buffer<T: Copy> {
    Host(Vec<T>),
    #[cfg(feature = "gpu")]
    Device(gpu_core::DeviceBuffer<T>),
}

impl Serialize for Buffer<f64> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let bytes = match self {
            Host(data) => {
                let mut bytes = Vec::with_capacity(data.len() * size_of::<f64>());
                for x in data {
                    for b in x.to_le_bytes() {
                        bytes.push(b);
                    }
                }
                bytes
            }

            #[cfg(feature = "gpu")]
            Device(_) => panic!(),
        };
        serializer.serialize_bytes(&bytes)
    }
}

#[derive(Clone, serde::Serialize)]
pub struct Patch {
    /// The region of index space covered by this patch.
    rect: Rectangle<i64>,

    /// The number of fields stored at each zone.
    num_fields: usize,

    /// The backing array of data on this patch.
    data: Buffer<f64>,
}

impl Patch {
    /// Generates a patch in host memory of zeros over the given index space.
    pub fn zeros<I: Clone + Into<IndexSpace>>(num_fields: usize, space: &I) -> Self {
        let space: IndexSpace = space.clone().into();
        let data = Host(vec![0.0; space.len() * num_fields]);

        Self {
            rect: space.into(),
            num_fields,
            data,
        }
    }

    /// Generates a patch in host memory covering the given space, with values
    /// defined from a closure.
    pub fn from_scalar_function<I, F>(space: &I, f: F) -> Self
    where
        I: Clone + Into<IndexSpace>,
        F: Fn((i64, i64)) -> f64,
    {
        Self::from_vector_function(space, |i| [f(i)])
    }

    /// Generates a patch in host memory covering the given space, with values
    /// defined from a closure which returns a fixed-length array. The number
    /// of fields in the patch is inferred from the size of the fixed length
    /// array returned by the closure.
    pub fn from_vector_function<I, F, const NUM_FIELDS: usize>(space: &I, f: F) -> Self
    where
        I: Clone + Into<IndexSpace>,
        F: Fn((i64, i64)) -> [f64; NUM_FIELDS],
    {
        Self::from_slice_function(space, NUM_FIELDS, |i, s| s.clone_from_slice(&f(i)))
    }

    /// Generates a patch in host memory covering the given space, with values
    /// defined from a closure which operates on mutable slices.
    pub fn from_slice_function<I, F>(space: &I, num_fields: usize, f: F) -> Self
    where
        I: Clone + Into<IndexSpace>,
        F: Fn((i64, i64), &mut [f64]),
    {
        let space: IndexSpace = space.clone().into();
        let mut data = vec![0.0; space.len() * num_fields];

        for (index, slice) in space.iter().zip(data.chunks_exact_mut(num_fields)) {
            f(index, slice)
        }
        Self {
            rect: space.into(),
            num_fields,
            data: Host(data),
        }
    }

    /// Returns the index space for this patch.
    pub fn index_space(&self) -> IndexSpace {
        IndexSpace::from(&self.rect)
    }

    pub fn rect(&self) -> Rectangle<i64> {
        self.rect.clone()
    }

    /// Returns the device where the data buffer lives, if it's a device
    /// buffer, and `None` otherwise.
    #[cfg(feature = "gpu")]
    pub fn device(&self) -> Option<Device> {
        match self.data {
            Device(ref data) => Some(data.device()),
            Host(_) => None,
        }
    }

    /// Returns an immutable pointer to the underlying storage. The pointer
    /// will reference data on the host or one of the GPU devices, depending
    /// on where the buffer resides.
    pub fn as_ptr(&self) -> *const f64 {
        match self.data {
            Host(ref data) => data.as_ptr(),
            #[cfg(feature = "gpu")]
            Device(ref data) => data.as_device_ptr(),
        }
    }

    /// Returns a mutable pointer to the underlying storage. The pointer will
    /// reference data on the host or one of the GPU devices, depending on
    /// where the buffer resides.
    pub fn as_mut_ptr(&mut self) -> *mut f64 {
        match self.data {
            Host(ref mut data) => data.as_mut_ptr(),
            #[cfg(feature = "gpu")]
            Device(ref mut data) => data.as_mut_device_ptr(),
        }
    }

    /// Makes a deep copy of this buffer on the given device. This buffer may
    /// reside on the host, or on any device.
    #[cfg(feature = "gpu")]
    pub fn to_device(&self, device: Device) -> Self {
        Self {
            rect: self.rect.clone(),
            num_fields: self.num_fields,
            data: match self.data {
                Host(ref data) => Device(device.buffer_from(data)),
                Device(ref data) => Device(data.copy_to(device)),
            },
        }
    }

    /// Makes a deep copy of this buffer on the given device, if necessary. If
    /// the buffer already resides on the given device, no memory transfers or
    /// copies will take place.
    #[cfg(feature = "gpu")]
    pub fn into_device(self, device: Device) -> Self {
        Self {
            rect: self.rect.clone(),
            num_fields: self.num_fields,
            data: match self.data {
                Host(data) => Device(device.buffer_from(&data)),
                Device(data) => Device(if data.device() == device {
                    data
                } else {
                    data.copy_to(device)
                }),
            },
        }
    }

    /// Makes a deep copy of this buffer to host memory. This buffer may
    /// reside on the host, or on any device.
    #[cfg(feature = "gpu")]
    pub fn to_host(&self) -> Self {
        Self {
            rect: self.rect.clone(),
            num_fields: self.num_fields,
            data: match self.data {
                Host(ref data) => Host(data.clone()),
                Device(ref data) => Host(data.to_vec()),
            },
        }
    }

    /// Makes a deep copy of this buffer to host memory, if necessary. If the
    /// buffer already resides on the host, no memory transfers or copies will
    /// take place.
    #[cfg(feature = "gpu")]
    pub fn into_host(self) -> Self {
        Self {
            rect: self.rect.clone(),
            num_fields: self.num_fields,
            data: match self.data {
                Host(data) => Host(data),
                Device(data) => Host(data.to_vec()),
            },
        }
    }

    /// Extracts a subset of this patch and returns it. This method panics if
    /// the given space is not fully contained within this patch.
    pub fn extract<I: Clone + Into<IndexSpace>>(&self, dst_space: &I) -> Self {
        let dst_space: IndexSpace = dst_space.clone().into();
        let src_space: IndexSpace = self.rect.clone().into();

        assert! {
            src_space.contains_space(&dst_space),
            "the index space is out of bounds"
        }

        match &self.data {
            Host(ref src_data) => {
                let mut dst_data = vec![0.0; dst_space.len() * self.num_fields];
                for index in dst_space.iter() {
                    let n_src = src_space.row_major_offset(index);
                    let n_dst = dst_space.row_major_offset(index);
                    dst_data[n_dst] = src_data[n_src];
                }
                Self {
                    rect: dst_space.into(),
                    num_fields: self.num_fields,
                    data: Host(dst_data),
                }
            }
            #[cfg(feature = "gpu")]
            Device(ref src_data) => {
                let start = [
                    src_space.start().0 as usize,
                    src_space.start().1 as usize,
                    0,
                ];
                let shape = [src_space.dim().0, src_space.dim().1, 1];
                let count = [dst_space.dim().0, dst_space.dim().1, 1];
                let dst_data = src_data.extract_3d(start, shape, count, self.num_fields);
                Self {
                    rect: dst_space.into(),
                    num_fields: self.num_fields,
                    data: Device(dst_data),
                }
            }
        }
    }

    /// Copies values from this patch into another one. The two patches must
    /// have the same number of fields, but they do not need to have the same
    /// index space. Only the elements at the overlapping part of the index
    /// spaces are copied; the non-overlapping part of the target patch is
    /// unchanged. Memory will be migrated across the host and device, or
    /// between devices as needed.
    pub fn copy_into(&self, target: &mut Self) {
        assert!(self.num_fields == target.num_fields);

        let overlap = self.index_space().intersect(target.index_space());
        let src_region = overlap.memory_region_in(self.index_space());
        let dst_region = overlap.memory_region_in(target.index_space());

        match (&self.data, &mut target.data) {
            (Host(ref src_data), Host(ref mut dst_data)) => src_region
                .iter_slice(src_data, self.num_fields)
                .zip(dst_region.iter_slice_mut(dst_data, self.num_fields))
                .for_each(|(s, d)| d.copy_from_slice(s)),

            #[cfg(feature = "gpu")]
            (Device(ref src_data), Device(ref mut dst_data)) => {
                let dst_start = [dst_region.start.0, dst_region.start.1, 0];
                let dst_shape = [dst_region.shape.0, dst_region.shape.1, 1];
                let dst_count = [dst_region.count.0, dst_region.count.1, 1];
                let src_start = [src_region.start.0, src_region.start.1, 0];
                let src_shape = [src_region.shape.0, src_region.shape.1, 1];
                let src_count = [src_region.count.0, src_region.count.1, 1];

                assert_eq!(src_count, dst_count);

                dst_data.memcpy_3d(dst_start, dst_shape, src_data, src_start, src_shape, src_count, self.num_fields);
            },

            #[cfg(feature = "gpu")]
            (Device(_), Host(_)) => {
                self.to_host().copy_into(target)
            }

            #[cfg(feature = "gpu")]
            (Host(_), Device(ref dst_data)) => {
                self.to_device(dst_data.device()).copy_into(target)
            }
        }
    }
}
