use std::ffi::{c_int, c_void};

use cudarc::driver::sys::CUstream_st;

unsafe extern "C" {
    pub fn launch_sin2(out: *mut c_void, inp: *const c_void, n: c_int, stream: &*mut CUstream_st);
}

#[cfg(test)]
mod tests {
    use cudarc::driver::{DevicePtr, DevicePtrMut, DriverError};

    #[test]
    fn test_simple() -> Result<(), DriverError> {
        let data: Vec<f32> = (0..100).map(|u| u as f32).collect();
        // Get a stream for GPU 0
        let dev = cudarc::driver::CudaDevice::new(0)?;

        // copy a rust slice to the device
        let inp = dev.htod_copy(data.clone())?;
        let mut out = dev.alloc_zeros::<f32>(100)?;

        let out_ptr = *out.device_ptr_mut() as *mut core::ffi::c_void;
        let inp_ptr = *inp.device_ptr() as *const core::ffi::c_void;
        unsafe { super::launch_sin2(out_ptr, inp_ptr, 100, dev.cu_stream()) };

        let out_host: Vec<f32> = dev.dtoh_sync_copy(&out)?;
        assert_eq!(out_host.len(), data.len());
        // Only approximations can be asserted
        let expected: Vec<_> = data.into_iter().map(f32::sin).collect();
        println!("Expect {expected:?}");
        println!("Got {out_host:?}");
        for (i, (l, r)) in out_host.into_iter().zip(expected.into_iter()).enumerate() {
            let diff = (l - r).abs() / (l + 1e-10);
            assert!(diff < 1e-3, "{l} != {r} (diff = {diff:?}, location = {i})");
        }
        Ok(())
    }
}
