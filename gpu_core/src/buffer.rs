//! Exports a `Buffer` enum which can represent a `Vec` or a `DeviceBuffer`.
use cfg_if::cfg_if;
use crate::Device;

#[cfg(feature = "gpu")]
use crate::DeviceBuffer;

#[derive(Clone)]
pub enum Buffer<T: Copy> {
    Host(Vec<T>),
    #[cfg(feature = "gpu")]
    Device(DeviceBuffer<T>),
}

impl<T: Copy> Buffer<T> {
    pub fn len(&self) -> usize {
       match self {
           Buffer::Host(data) => data.len(),
           #[cfg(feature = "gpu")]
           Buffer::Device(data) => data.len(),
       } 
    }

    /// Returns the device where the data buffer lives, if it's a device
    /// buffer, and `None` otherwise.
    pub fn device(&self) -> Option<Device> {
        match self {
            Buffer::Host(_) => None,
            #[cfg(feature = "gpu")]
            Buffer::Device(data) => Some(data.device()),
        }
    }

    /// Returns the underlying data as a slice, if it lives on the host,
    /// otherwise returns `None`.
    pub fn as_slice(&self) -> Option<&[T]> {
        match self {
            Buffer::Host(data) => Some(data.as_slice()),
            #[cfg(feature = "gpu")]
            Buffer::Device(_) => None,
        }
    }

    /// Returns the underlying data as a device buffer, if it lives on a
    /// device, otherwise returns `None`.
    #[cfg(feature = "gpu")]
    pub fn as_device_buffer(&self) -> Option<&DeviceBuffer<T>> {
        match self {
            Buffer::Host(_) => None,
            Buffer::Device(data) => Some(data),
        }
    }

    /// Returns an immutable pointer to the underlying storage. The pointer
    /// will reference data on the host or one of the GPU devices, depending
    /// on where the buffer resides.
    pub fn as_ptr(&self) -> *const T {
        match self {
            Buffer::Host(data) => data.as_ptr(),
            #[cfg(feature = "gpu")]
            Buffer::Device(data) => data.as_device_ptr(),
        }
    }

    /// Returns a mutable pointer to the underlying storage. The pointer will
    /// reference data on the host or one of the GPU devices, depending on
    /// where the buffer resides.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        match self {
            Buffer::Host(ref mut data) => data.as_mut_ptr(),
            #[cfg(feature = "gpu")]
            Buffer::Device(ref mut data) => data.as_mut_device_ptr(),
        }
    }

    /// Makes a deep copy of this buffer on the given device. This buffer may
    /// reside on the host, or on any device. This function will panic if GPU
    /// support is not available.
    pub fn to_device(&self, device: Device) -> Self {
        cfg_if! {
            if #[cfg(feature = "gpu")] {
                match self {
                    Buffer::Host(data) => Buffer::Device(device.buffer_from(data)),
                    Buffer::Device(data) => Buffer::Device(data.copy_to(device)),
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
                match self {
                    Buffer::Host(data) => Buffer::Device(device.buffer_from(&data)),
                    Buffer::Device(data) => Buffer::Device(if data.device() == device {
                        data
                    } else {
                        data.copy_to(device)
                    })
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
                match self {
                    Buffer::Host(data) => Buffer::Host(data.clone()),
                    Buffer::Device(data) => Buffer::Host(data.to_vec()),
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
                match self {
                    Buffer::Host(data) => Buffer::Host(data),
                    Buffer::Device(data) => Buffer::Host(data.to_vec()),
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
}
