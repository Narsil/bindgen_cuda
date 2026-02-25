mod kernel;

#[cfg(test)]
mod tests {
    use cudarc::driver::{DriverError, LaunchConfig, PushKernelArg};

    #[test]
    fn test_simple() -> Result<(), DriverError> {
        let data: Vec<f32> = (0..100).map(|u| u as f32).collect();
        // Get a stream for GPU 0
        let ctx = cudarc::driver::CudaContext::new(0)?;
        let stream = ctx.default_stream();

        // copy a rust slice to the device
        let inp = stream.memcpy_stod(&data)?;

        // or allocate directly
        let mut out = stream.alloc_zeros::<f32>(100)?;
        // Dynamically load it into the device
        let ptx = cudarc::nvrtc::Ptx::from_src(crate::kernel::CUDA);
        let module = ctx.load_module(ptx)?;
        let sin_kernel = module.load_function("sin_kernel")?;
        let mut builder = stream.launch_builder(&sin_kernel);
        builder.arg(&mut out);
        builder.arg(&inp);
        builder.arg(&100usize);
        unsafe { builder.launch(LaunchConfig::for_num_elems(100)) }?;
        let out_host: Vec<f32> = stream.memcpy_dtov(&out)?;

        assert_eq!(out_host.len(), data.len());
        // Only approximations can be asserted
        let expected: Vec<_> = data.into_iter().map(f32::sin).collect();
        for (i, (l, r)) in out_host.into_iter().zip(expected.into_iter()).enumerate() {
            let diff = (l - r).abs() / (l + 1e-10);
            assert!(diff < 1e-3, "{l} != {r} (diff = {diff:?}, location = {i})");
        }
        Ok(())
    }
}
