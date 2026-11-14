# Configure with -DUSE_EXECUTORCH=ON to run exported/policy.pte with ExecuTorch instead of
# exported/policy.onnx with ONNX Runtime. The prebuilt library ships in the executorch Python package
# and needs CMake 3.28 or newer.
option(USE_EXECUTORCH "Run the policy with ExecuTorch" OFF)
if(USE_EXECUTORCH)
  find_package(executorch 1.5 REQUIRED CONFIG COMPONENTS runtime kernels_optimized backend_xnnpack)
  add_compile_definitions(USE_EXECUTORCH)
  link_libraries(executorch::runtime executorch::kernels_optimized executorch::backend_xnnpack)
endif()
