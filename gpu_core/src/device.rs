use once_cell::sync::Lazy;
use std::sync::Mutex;
use std::marker::PhantomData;

static SYSTEM: Lazy<Mutex<Option<System>>> = Lazy::new(|| Mutex::new(Some(System::create())));
static FREE_DEVICE_IDS: Lazy<Mutex<Vec<usize>>> = Lazy::new(|| Mutex::new((0..8).collect()));

pub struct System {}

impl System {
    fn create() -> Self {
        Self {}
    }
    pub fn take() -> Option<Self> {
        SYSTEM.lock().unwrap().take()
    }
    pub fn take_device(&self) -> Option<Device> {
        FREE_DEVICE_IDS
            .lock()
            .unwrap()
            .pop()
            .map(|id| Device { id, _system: PhantomData })
    }
}

impl Drop for System {
    fn drop(&mut self) {
        SYSTEM.lock().unwrap().insert(System::create());
    }
}

pub struct Device<'a> {
    pub id: usize,
    _system: PhantomData<&'a System>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name() {
        let system = System::take().unwrap();
        let gpu1 = system.take_device();
        let gpu2 = system.take_device();

        drop(gpu2);
        drop(gpu1);
        drop(system);
    }
}
