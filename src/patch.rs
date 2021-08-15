use cfg_if::cfg_if;
use gpu_core::{Buffer, Device};
use gridiron::index_space::IndexSpace;
use gridiron::rect_map::Rectangle;
use Buffer::*;

mod serde_buffer {
    use gpu_core::Buffer;
    use serde::de::Visitor;
    use std::fmt;
    use std::mem::size_of;
    use Buffer::*;

    struct BufferVisitor;

    impl<'de> Visitor<'de> for BufferVisitor {
        type Value = Buffer<f64>;
        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a buffer of bytes")
        }
        fn visit_bytes<E>(self, v: &[u8]) -> Result<Self::Value, E> {
            let data = v
                .chunks_exact(8)
                .map(|c| [c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                .map(f64::from_le_bytes)
                .collect();
            Ok(Host(data))
        }
    }

    pub fn serialize<S: serde::Serializer>(
        buffer: &Buffer<f64>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        let byte_vec = |data: &[f64]| {
            let mut bytes = Vec::with_capacity(data.len() * size_of::<f64>());
            for x in data {
                for b in x.to_le_bytes() {
                    bytes.push(b);
                }
            }
            bytes
        };
        let bytes = match buffer {
            Host(data) => byte_vec(data),
            #[cfg(feature = "gpu")]
            Device(data) => byte_vec(&data.to_vec()),
        };
        serializer.serialize_bytes(&bytes)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Buffer<f64>, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_bytes(BufferVisitor)
    }
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Patch {
    /// The region of index space covered by this patch.
    rect: Rectangle<i64>,

    /// The number of fields stored at each zone.
    num_fields: usize,

    /// The backing array of data on this patch.
    #[serde(with = "serde_buffer")]
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
        self.data.device()
    }

    /// Returns the underlying data as a slice, if it lives on the host,
    /// otherwise returns `None`.
    pub fn as_slice(&self) -> Option<&[f64]> {
        self.data.as_slice()
    }

    /// Returns the underlying data as a device buffer, if it lives on a
    /// device, otherwise returns `None`.
    #[cfg(feature = "gpu")]
    pub fn as_device_buffer(&self) -> Option<&gpu_core::DeviceBuffer<f64>> {
        self.data.as_device_buffer()
    }

    /// Returns an immutable pointer to the underlying storage. The pointer
    /// will reference data on the host or one of the GPU devices, depending
    /// on where the buffer resides.
    pub fn as_ptr(&self) -> *const f64 {
        self.data.as_ptr()
    }

    /// Returns a mutable pointer to the underlying storage. The pointer will
    /// reference data on the host or one of the GPU devices, depending on
    /// where the buffer resides.
    pub fn as_mut_ptr(&mut self) -> *mut f64 {
        self.data.as_mut_ptr()
    }

    /// Makes a deep copy of this buffer on the given device. This buffer may
    /// reside on the host, or on any device. This function will panic if GPU
    /// support is not available.
    pub fn to_device(&self, device: Device) -> Self {
        cfg_if! {
            if #[cfg(feature = "gpu")] {
                Self {
                    rect: self.rect.clone(),
                    num_fields: self.num_fields,
                    data: self.data.to_device(device),
                }
            } else {
                std::convert::identity(device); // black-box
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
                    data: self.data.into_device(device),
                }
            } else {
                std::convert::identity(device); // black-box
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
                    data: self.data.to_host(),
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
                    data: self.data.into_host(),
                }
            } else {
                self
            }
        }
    }

    /// Consumes this buffer and ensures it resides the given device, if it's
    /// `Some`. Otherwise if `device` is `None` then ensure this buffer
    /// resides in host memory.
    pub fn on(self, device: Option<Device>) -> Self {
        if let Some(device) = device {
            self.into_device(device)
        } else {
            self.into_host()
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
    /// host, or between devices as needed. This method panics if the source
    /// and destination index spaces do not overlap.
    pub fn copy_into(&self, target: &mut Self) {
        assert!(self.num_fields == target.num_fields);

        let overlap = self
            .index_space()
            .intersect(&target.index_space())
            .expect("source and destination index spaces do not overlap");
        let src_reg = overlap.memory_region_in(&self.index_space());
        let dst_reg = overlap.memory_region_in(&target.index_space());
        let nq = self.num_fields;

        if self.device() != target.device() {
            return self.extract(&overlap).on(target.device()).copy_into(target);
        }

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
                )
            }

            #[cfg(feature = "gpu")]
            _ => unreachable!(),
        }
    }

    pub fn map_mut<F>(&mut self, subset: &IndexSpace, f: F)
    where
        F: Fn((i64, i64), &mut [f64]),
    {
        Self::from_slice_function(subset, self.num_fields, f).copy_into(self);
    }

    /// Returns a new patch, upsampled to twice the resolution as this one.
    /// The new index space is scaled by a factor of two. No interpolation is
    /// performed; the new patch has uniform values within 2x2 cell regions.
    /// The resulting patch resides on the same device as this one, but the
    /// upsampling is done on the CPU, so, invoked on device patches, this
    /// function bears the penalty of the dreaded D -> H -> D round trip.
    pub fn upsample(&self) -> Self {
        match &self.data {
            Host(ref data) => {
                let (_i, nj) = self.index_space().dim();
                let (i0, j0) = self.index_space().start();
                let nq = self.num_fields;
                let si = nq * nj;
                let sj = nq;
                let space = self.index_space().refine_by(2);
                Self::from_slice_function(&space, nq, |(i, j), slice| {
                    let n = (i / 2 - i0) as usize * si + (j / 2 - j0) as usize * sj;
                    slice.copy_from_slice(&data[n..n + nq])
                })
            }
            #[cfg(feature = "gpu")]
            Device(ref data) => self.to_host().upsample().to_device(data.device()),
        }
    }

    /// Convenience method to upsample this patch in-place.
    pub fn upsample_mut(&mut self) {
        *self = self.upsample()
    }
}

#[cfg(feature = "gpu")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh;
    use crate::sailfish::StructuredMesh;
    use gridiron::index_space::{range2d, Axis};

    #[test]
    fn copy_patch_subset_from_host_to_device() {
        for device in gpu_core::all_devices() {
            let src_space = range2d(10..20, 0..200);
            let dst_space = range2d(0..100, 0..200);
            let src = Patch::from_vector_function(&src_space, |(i, j)| [i as f64, j as f64]);
            let mut dst = Patch::zeros(2, &dst_space).into_device(device);
            src.copy_into(&mut dst);
            assert_eq!(
                src.into_host().as_slice(),
                dst.extract(&src_space).into_host().as_slice()
            );
        }
    }

    #[test]
    fn fill_guard_regions_on_host() {
        fill_guard_regions_impl(None)
    }

    #[test]
    fn fill_guard_regions_on_device() {
        fill_guard_regions_impl(Device::with_id(0))
    }

    fn fill_guard_regions_impl(device: Option<Device>) {
        let setup = |(i, j)| [i as f64, j as f64, 0.0];

        let global_structured_mesh = StructuredMesh::centered_square(10.0, 1024);
        let local_space = range2d(0..256, 0..256);
        let primitive = Patch::from_vector_function(&local_space, setup);

        let local_space = primitive.index_space();
        let local_space_ext = local_space.extend_all(2);
        let global_mesh = mesh::Mesh::Structured(global_structured_mesh);
        let global_space_ext = global_mesh.index_space().extend_all(2);

        let guard_spaces = [
            global_space_ext.keep_lower(2, Axis::I),
            global_space_ext.keep_upper(2, Axis::I),
            global_space_ext.keep_lower(2, Axis::J),
            global_space_ext.keep_upper(2, Axis::J),
        ];

        let mut primitive1 = Patch::zeros(3, &local_space.extend_all(2)).on(device);
        primitive.copy_into(&mut primitive1);

        for space in guard_spaces {
            if let Some(overlap) = space.intersect(&local_space_ext) {
                println!("{:?}", overlap);
                Patch::from_vector_function(&overlap, setup).copy_into(&mut primitive1)
            }
        }

        assert_eq!(
            primitive1.extract(&local_space).into_host().as_slice(),
            primitive.as_slice()
        );
    }
}
