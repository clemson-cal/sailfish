pub mod ffi {
    pub(super) static BUFFER_MODE_HOST: i32 = 1;

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub(crate) struct Patch
    {
        pub(super) start: [i32; 2],
        pub(super) count: [i32; 2],
        pub(super) jumps: [i32; 2],
        pub(super) num_fields: i32,
        pub(super) buffer_mode: i32,
        pub(super) data: *mut f64,
    }

    extern "C" {
        pub(super) fn patch_new(
            start_i: i32,
            start_j: i32,
            count_i: i32,
            count_j: i32,
            num_fields: i32,
            buffer_mode: i32,
            data: *const f64) -> Patch;
        pub(super) fn patch_del(patch: Patch);
        pub(super) fn patch_set(patch: Patch, i: i32, j: i32, q: i32, y: f64);
        // pub(super) fn patch_get(patch: Patch, i: i32, j: i32, q: i32) -> f64;
        pub(super) fn patch_contains(patch: Patch, other: Patch) -> i32;
        pub(super) fn patch_clone(patch: Patch) -> Patch;
    }    

    #[cfg(feature = "cuda")]
    extern "C" {
        pub(super) fn patch_clone_to_device(patch: Patch) -> Patch;
        pub(super) fn patch_clone_to_host(patch: Patch) -> Patch;
    }
}

pub mod host {
    use super::ffi;
    #[cfg(feature = "cuda")]
    use super::device;

    pub struct Patch(pub(crate) ffi::Patch);

    impl Patch {
        pub fn from_fn<F, const N: usize>(start: [i32; 2], count: [u32; 2], f: F) -> Self 
        where
            F: Fn(i32, i32) -> [f64; N]
        {
            Self(unsafe {
                let c = ffi::patch_new(
                    start[0],
                    start[1],
                    count[0] as i32,
                    count[1] as i32,
                    N as i32,
                    ffi::BUFFER_MODE_HOST,
                    std::ptr::null(),
                );
                for i in start[0]..start[0] + count[0] as i32 {
                    for j in start[1]..start[1] + count[1] as i32 {
                        let x = f(i, j);
                        for q in 0..N {
                            ffi::patch_set(c, i, j, q as i32, x[q]);
                        }
                    }
                }
                c
            })
        }

        pub fn zeros(start: [i32; 2], count: [u32; 2], num_fields: u32) -> Self {
            Self(unsafe {
                let c = ffi::patch_new(
                    start[0],
                    start[1],
                    count[0] as i32,
                    count[1] as i32,
                    num_fields as i32,
                    ffi::BUFFER_MODE_HOST,
                    std::ptr::null(),
                );
                for i in start[0]..start[0] + count[0] as i32 {
                    for j in start[1]..start[1] + count[1] as i32 {
                        for q in 0..num_fields {
                            ffi::patch_set(c, i, j, q as i32, 0.0);
                        }
                    }
                }
                c
            })
        }

        pub fn from_vec(start: [i32; 2], count: [u32; 2], num_fields: u32, data: &Vec<f64>) -> Self {
            assert!{
                data.len() == (count[0] * count[1] * num_fields) as usize,
                "input data has the wrong size for the given count"
            };
            Self(unsafe {
                let c = ffi::patch_new(
                    start[0],
                    start[1],
                    count[0] as i32,
                    count[1] as i32,
                    num_fields as i32,
                    ffi::BUFFER_MODE_HOST,
                    std::ptr::null(),
                );
                std::ptr::copy_nonoverlapping(data.as_ptr(), c.data, data.len());
                c
            })
        }

        pub fn start(&self) -> [i32; 2] {
            [self.0.start[0] as i32, self.0.start[1] as i32]
        }

        pub fn count(&self) -> [u32; 2] {
            [self.0.count[0] as u32, self.0.count[1] as u32]
        }

        pub fn num_fields(&self) -> u32 {
            self.0.num_fields as u32
        }

        pub fn contains(&self, other: &Self) -> bool {
            unsafe {
                ffi::patch_contains(self.0, other.0) != 0
            }
        }

        #[cfg(feature = "cuda")]
        pub fn to_device(&self) -> device::Patch {
            unsafe {
                device::Patch(ffi::patch_clone_to_device(self.0))
            }
        }

        pub fn to_vec(&self) -> Vec<f64> {
            let [ni, nj] = self.count();
            let mut res = vec![0.0; (ni * nj * self.num_fields()) as usize];
            unsafe {
                std::ptr::copy_nonoverlapping(self.0.data, res.as_mut_ptr(), res.len());
            }
            res
        }
    }

    impl Clone for Patch {
        fn clone(&self) -> Self {
            Self(unsafe {
                ffi::patch_clone(self.0)
            })
        }
    }

    impl Drop for Patch {
        fn drop(&mut self) {
            unsafe {
                ffi::patch_del(self.0)
            }
        }
    }
}

#[cfg(feature = "cuda")]
pub mod device {
    use super::{ffi, host};

    pub struct Patch(pub(crate) ffi::Patch);

    impl Patch {
        pub fn start(&self) -> [usize; 2] {
            [self.0.start[0] as usize, self.0.start[1] as usize]
        }

        pub fn count(&self) -> [usize; 2] {
            [self.0.count[0] as usize, self.0.count[1] as usize]
        }

        pub fn num_fields(&self) -> usize {
            self.0.num_fields as usize
        }

        pub fn contains(&self, other: &Self) -> bool {
            unsafe {
                ffi::patch_contains(self.0, other.0) != 0
            }
        }

        pub fn to_host(&self) -> host::Patch {
            unsafe {
                host::Patch(ffi::patch_clone_to_host(self.0))
            }
        }
    }

    impl Drop for Patch {
        fn drop(&mut self) {
            unsafe {
                ffi::patch_del(self.0)
            }
        }
    }
}
