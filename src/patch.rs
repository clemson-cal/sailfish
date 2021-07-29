use gpu_core::Device;
use gridiron::index_space::IndexSpace;
use gridiron::rect_map::Rectangle;
use serde::ser::{Serialize, Serializer};
use std::mem::size_of;

#[derive(Clone)]
enum Buffer<T: Copy> {
    Host(Vec<T>),
    Device(gpu_core::DeviceBuffer<T>),
}

impl Serialize for Buffer<f64> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let bytes = match self {
            Buffer::Host(data) => {
                let mut bytes = Vec::with_capacity(data.len() * size_of::<f64>());
                for x in data {
                    for b in x.to_le_bytes() {
                        bytes.push(b);
                    }
                }
                bytes
            }
            Buffer::Device(_) => panic!(),
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
    /// Creates a new empty patch.
    // pub fn new() -> Self {
    //     Self {
    //         rect: (0..0, 0..0),
    //         num_fields: 0,
    //         data: Buffer::Host(Vec::new()),
    //     }
    // }

    /// Generates a patch of zeros over the given index space.
    pub fn zeros<I: Clone + Into<IndexSpace>>(num_fields: usize, space: &I) -> Self {
        let space: IndexSpace = space.clone().into();
        let data = Buffer::Host(vec![0.0; space.len() * num_fields]);

        Self {
            rect: space.into(),
            num_fields,
            data,
        }
    }

    /// Generates a patch covering the given space, with values defined from a
    /// closure.
    pub fn from_scalar_function<I, F>(space: &I, f: F) -> Self
    where
        I: Clone + Into<IndexSpace>,
        F: Fn((i64, i64)) -> f64,
    {
        Self::from_vector_function(space, |i| [f(i)])
    }

    /// Generates a patch covering the given space, with values defined from a
    /// closure which returns a fixed-length array. The number of fields in
    /// the patch is inferred from the size of the fixed length array returned
    /// by the closure.
    pub fn from_vector_function<I, F, const NUM_FIELDS: usize>(space: &I, f: F) -> Self
    where
        I: Clone + Into<IndexSpace>,
        F: Fn((i64, i64)) -> [f64; NUM_FIELDS],
    {
        Self::from_slice_function(space, NUM_FIELDS, |i, s| s.clone_from_slice(&f(i)))
    }

    /// Generates a patch covering the given space, with values defined from a
    /// closure which operates on mutable slices.
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
            data: Buffer::Host(data),
        }
    }

    /// Returns the index space for this patch.
    pub fn index_space(&self) -> IndexSpace {
        IndexSpace::from(&self.rect)
    }

    /// Makes a deep copy of this buffer on the given device. This buffer may
    /// reside on the host, or on any device.
    pub fn to_device(&self, device: Device) -> Self {
        Self {
            rect: self.rect.clone(),
            num_fields: self.num_fields,
            data: match self.data {
                Buffer::Host(ref data) => Buffer::Device(device.buffer_from(data)),
                Buffer::Device(ref data) => Buffer::Device(data.copy_to(device)),
            },
        }
    }

    /// Makes a deep copy of this buffer on the given device, if necessary. If
    /// the buffer already resides on the given device, no memory transfers or
    /// copies will take place.
    pub fn into_device(self, device: Device) -> Self {
        Self {
            rect: self.rect.clone(),
            num_fields: self.num_fields,
            data: match self.data {
                Buffer::Host(data) => Buffer::Device(device.buffer_from(&data)),
                Buffer::Device(data) => Buffer::Device(if data.device() == device {
                    data
                } else {
                    data.copy_to(device)
                }),
            },
        }
    }

    /// Makes a deep copy of this buffer to host memory. This buffer may
    /// reside on the host, or on any device.
    pub fn to_host(&self) -> Self {
        Self {
            rect: self.rect.clone(),
            num_fields: self.num_fields,
            data: match self.data {
                Buffer::Host(ref data) => Buffer::Host(data.clone()),
                Buffer::Device(ref data) => Buffer::Host(data.to_vec()),
            },
        }
    }

    /// Makes a deep copy of this buffer to host memory, if necessary. If the
    /// buffer already resides on the host, no memory transfers or copies will
    /// take place.
    pub fn into_host(self) -> Self {
        Self {
            rect: self.rect.clone(),
            num_fields: self.num_fields,
            data: match self.data {
                Buffer::Host(data) => Buffer::Host(data),
                Buffer::Device(data) => Buffer::Host(data.to_vec()),
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
            Buffer::Host(ref src_data) => {
                let mut dst_data = vec![0.0; dst_space.len() * self.num_fields];
                for index in dst_space.iter() {
                    let n_src = src_space.row_major_offset(index);
                    let n_dst = dst_space.row_major_offset(index);
                    dst_data[n_dst] = src_data[n_src];
                }
                Self {
                    rect: dst_space.into(),
                    num_fields: self.num_fields,
                    data: Buffer::Host(dst_data),
                }
            }
            Buffer::Device(ref src_data) => {
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
                    data: Buffer::Device(dst_data),
                }
            }
        }
    }
}
