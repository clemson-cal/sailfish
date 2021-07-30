#[cfg(feature = "gpu")]
use cfg_if::cfg_if;
use gpu_core::Device;
use gridiron::index_space::IndexSpace;
use gridiron::rect_map::Rectangle;
use serde::de::{Deserialize, Deserializer};
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

impl<'de> Deserialize<'de> for Buffer<f64> {
    fn deserialize<D>(_deserializer: D) -> Result<Buffer<f64>, D::Error>
    where
        D: Deserializer<'de>,
    {
        todo!()
    }
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
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
    pub fn zeros(num_fields: usize, space: &IndexSpace) -> Self {
        Self {
            rect: space.into(),
            num_fields,
            data: Host(vec![0.0; space.len() * num_fields]),
        }
    }

    /// Generates a patch in host memory covering the given space, with values
    /// defined from a closure.
    pub fn from_scalar_function<F>(space: &IndexSpace, f: F) -> Self
    where
        F: Fn((i64, i64)) -> f64,
    {
        Self::from_vector_function(space, |i| [f(i)])
    }

    /// Generates a patch in host memory covering the given space, with values
    /// defined from a closure which returns a fixed-length array. The number
    /// of fields in the patch is inferred from the size of the fixed length
    /// array returned by the closure.
    pub fn from_vector_function<F, const NUM_FIELDS: usize>(space: &IndexSpace, f: F) -> Self
    where
        F: Fn((i64, i64)) -> [f64; NUM_FIELDS],
    {
        Self::from_slice_function(space, NUM_FIELDS, |i, s| s.clone_from_slice(&f(i)))
    }

    /// Generates a patch in host memory covering the given space, with values
    /// defined from a closure which operates on mutable slices.
    pub fn from_slice_function<F>(space: &IndexSpace, num_fields: usize, f: F) -> Self
    where
        F: Fn((i64, i64), &mut [f64]),
    {
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
        self.rect.clone().into()
    }

    /// Returns the rectangle for this patch.
    pub fn rect(&self) -> Rectangle<i64> {
        self.rect.clone()
    }

    /// Returns the device where the data buffer lives, if it's a device
    /// buffer, and `None` otherwise.
    pub fn device(&self) -> Option<Device> {
        match self.data {
            Host(_) => None,
            #[cfg(feature = "gpu")]
            Device(ref data) => Some(data.device()),
        }
    }

    /// Returns the underlying data as a slice, if it lives on the host,
    /// otherwise returns `None`.
    pub fn as_slice(&self) -> Option<&[f64]> {
        match self.data {
            Host(ref data) => Some(data.as_slice()),
            #[cfg(feature = "gpu")]
            Device(_) => None,
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
    /// reside on the host, or on any device. This function will panic if GPU
    /// support is not available.
    #[cfg(feature = "gpu")]
    pub fn to_device(&self, device: Device) -> Self {
        cfg_if! {
            if #[cfg(feature = "gpu")] {
                Self {
                    rect: self.rect.clone(),
                    num_fields: self.num_fields,
                    data: match self.data {
                        Host(ref data) => Device(device.buffer_from(data)),
                        Device(ref data) => Device(data.copy_to(device)),
                    },
                }
            } else {
                unimplemented!("Patch::to_device requires gpu feature")
            }
        }
    }

    /// Makes a deep copy of this buffer on the given device, if necessary. If
    /// the buffer already resides on the given device, no memory transfers or
    /// copies will take place. This function will panic if GPU support is not
    /// available.
    pub fn into_device(self, device: Device) -> Self {
        cfg_if! {
            if #[cfg(feature = "gpu")] {
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
            } else {
                unimplemented!("Patch::into_device requires gpu feature")
            }
        }
    }

    /// Makes a deep copy of this buffer to host memory. This buffer may
    /// reside on the host, or on any device.
    pub fn to_host(&self) -> Self {
        cfg_if! {
            if #[cfg(feature = "gpu")] {
                Self {
                    rect: self.rect.clone(),
                    num_fields: self.num_fields,
                    data: match self.data {
                        Host(ref data) => Host(data.clone()),
                        Device(ref data) => Host(data.to_vec()),
                    },
                }
            } else {
                self.clone()
            }
        }
    }

    /// Makes a deep copy of this buffer to host memory, if necessary. If the
    /// buffer already resides on the host, no memory transfers or copies will
    /// take place.
    pub fn into_host(self) -> Self {
        cfg_if! {
            if #[cfg(feature = "gpu")] {
                Self {
                    rect: self.rect.clone(),
                    num_fields: self.num_fields,
                    data: match self.data {
                        Host(data) => Host(data),
                        Device(data) => Host(data.to_vec()),
                    },
                }
            } else {
                self
            }
        }
    }

    /// Extracts a subset of this patch and returns it, with memory residing
    /// in the same location as this buffer. This method panics if the given
    /// space is not fully contained within this patch.
    pub fn extract(&self, dst_space: &IndexSpace) -> Self {
        assert! {
            self.index_space().contains_space(&dst_space),
            "the index space is out of bounds"
        }

        match &self.data {
            Host(_) => {
                let mut result = Patch::zeros(self.num_fields, dst_space);
                self.copy_into(&mut result);
                result
            }

            #[cfg(feature = "gpu")]
            Device(ref src_data) => {
                let mut result = Self {
                    rect: dst_space.into(),
                    num_fields: self.num_fields,
                    data: Device(unsafe {
                        src_data
                            .device()
                            .uninit_buffer(dst_space.len() * self.num_fields)
                    }),
                };
                self.copy_into(&mut result);
                result
            }
        }
    }

    /// Copies values from this patch into another one. The two patches must
    /// have the same number of fields, but they do not need to have the same
    /// index space. Only the elements at the overlapping part of the index
    /// spaces are copied; the non-overlapping part of the target patch is
    /// unchanged. Memory will be migrated from host to device, device to
    /// host, or between devices as needed.
    pub fn copy_into(&self, target: &mut Self) {
        assert!(self.num_fields == target.num_fields);

        let overlap = self.index_space().intersect(target.index_space());
        let src_reg = overlap.memory_region_in(self.index_space());
        let dst_reg = overlap.memory_region_in(target.index_space());
        let nq = self.num_fields;

        match (&self.data, &mut target.data) {
            (Host(ref src), Host(ref mut dst)) => src_reg
                .iter_slice(src, nq)
                .zip(dst_reg.iter_slice_mut(dst, nq))
                .for_each(|(s, d)| d.copy_from_slice(s)),

            #[cfg(feature = "gpu")]
            (Device(ref src), Device(ref mut dst)) => {
                let dst_start = [dst_reg.start.0, dst_reg.start.1, 0];
                let dst_shape = [dst_reg.shape.0, dst_reg.shape.1, 1];
                let dst_count = [dst_reg.count.0, dst_reg.count.1, 1];
                let src_start = [src_reg.start.0, src_reg.start.1, 0];
                let src_shape = [src_reg.shape.0, src_reg.shape.1, 1];
                let src_count = [src_reg.count.0, src_reg.count.1, 1];

                assert_eq!(src_count, dst_count);

                dst.memcpy_3d(
                    dst_start, dst_shape, src, src_start, src_shape, src_count, nq,
                );
            }

            #[cfg(feature = "gpu")]
            (Device(_), Host(_)) => self.to_host().copy_into(target),

            #[cfg(feature = "gpu")]
            (Host(_), Device(ref dst)) => self.to_device(dst.device()).copy_into(target),
        }
    }
}
