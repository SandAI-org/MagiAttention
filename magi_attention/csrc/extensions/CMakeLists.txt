cmake_minimum_required(VERSION 3.18)

# Accept target CUDA architectures from the command line via -DMAGI_CUDA_ARCHITECTURES="90;100".
# Default to sm_90 if not specified.
if(NOT DEFINED MAGI_CUDA_ARCHITECTURES OR MAGI_CUDA_ARCHITECTURES STREQUAL "")
    set(MAGI_CUDA_ARCHITECTURES "90")
endif()
message(STATUS "[magi_attn_ext] MAGI_CUDA_ARCHITECTURES: '${MAGI_CUDA_ARCHITECTURES}'")

project(magi_attn_ext LANGUAGES CXX CUDA)

# ==============================================================================
# Dependency Configuration
# ==============================================================================

# Find PyTorch
# Tip: If Torch is not found, set CMAKE_PREFIX_PATH to your torch installation.
# e.g., run: python -c "import torch; print(torch.utils.cmake_prefix_path)"
find_package(Torch REQUIRED)

# PyTorch's find_package(Torch) / Caffe2Config.cmake injects -gencode flags for ALL
# architectures it was built with into CMAKE_CUDA_FLAGS. This causes compilation
# failures when the current CUDA toolkit doesn't support some of those architectures
# (e.g. compute_50 on CUDA 13). Strip them out and use our own.
message(STATUS "[magi_attn_ext] CMAKE_CUDA_FLAGS (from Torch): '${CMAKE_CUDA_FLAGS}'")
string(REGEX REPLACE "-gencode;arch=[^;]*,code=[^;]*;?" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
string(REGEX REPLACE "-gencode arch=[^ ]+" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
string(STRIP "${CMAKE_CUDA_FLAGS}" CMAKE_CUDA_FLAGS)
message(STATUS "[magi_attn_ext] CMAKE_CUDA_FLAGS (after cleanup): '${CMAKE_CUDA_FLAGS}'")

# Find Python 3 (Interpreter and Development libraries)
# We require the Development component to build C/C++ extensions.
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# ==============================================================================
# Source Management
# ==============================================================================

# Collect source files via glob (matches structure in setup.py)
# Note: In production CMake, listing files explicitly is often preferred to
# ensure build system updates when files are added, but GLOB works for extensions.
file(GLOB SOURCES
    "*.cpp"
    "*.cu"
)

# ==============================================================================
# Target Definition
# ==============================================================================

# Define the library as a Python module.
# "MODULE" tells CMake this is a loadable plugin (not a standard shared lib).
# "WITH_SOABI" automatically appends the correct Python extension suffix
# (e.g., .cpython-312-x86_64-linux-gnu.so) and removes the 'lib' prefix.
python3_add_library(magi_attn_ext MODULE ${SOURCES} WITH_SOABI)

# Set Language Standards
set_target_properties(magi_attn_ext PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

# Configure CUDA Architectures
# PyTorch sets CUDA_ARCHITECTURES to OFF on targets. We override it with our
# own list from MAGI_CUDA_ARCHITECTURES so only the desired archs are compiled.
set_target_properties(magi_attn_ext PROPERTIES
    CUDA_ARCHITECTURES "${MAGI_CUDA_ARCHITECTURES}"
)
get_target_property(_cuda_archs magi_attn_ext CUDA_ARCHITECTURES)
message(STATUS "[magi_attn_ext] Target CUDA_ARCHITECTURES property: '${_cuda_archs}'")

# Output Directory Configuration
# Places the compiled library two levels up from the source dir (likely into the package root).
set_target_properties(magi_attn_ext PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../"
)

# ==============================================================================
# Includes and Linking
# ==============================================================================

# Add include directories
# Aligns with setup.py: includes current source and the sibling 'common' directory.
target_include_directories(magi_attn_ext PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

# Link against PyTorch and Python libraries
# LibTorch provides most dependencies (CUDA runtime, cuBLAS, etc.)

# Find torch_python library
find_library(TORCH_PYTHON_LIBRARY torch_python
    HINTS
        "${TORCH_INSTALL_PREFIX}/lib"
        "/usr/local/lib/python3.12/dist-packages/torch/lib"
    PATHS
        "${CMAKE_PREFIX_PATH}/lib"
)

if(NOT TORCH_PYTHON_LIBRARY)
    message(WARNING "torch_python library not found. Linking may fail.")
else()
    message(STATUS "Found torch_python: ${TORCH_PYTHON_LIBRARY}")
endif()

target_link_libraries(magi_attn_ext PRIVATE
    ${TORCH_LIBRARIES}
    Python3::Python
    ${TORCH_PYTHON_LIBRARY}
)

# ==============================================================================
# Compilation Options
# ==============================================================================

# Define compiler-specific flags
target_compile_options(magi_attn_ext PRIVATE
    # C++ Host Compiler Flags
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3
        -fPIC
        -Wno-unused-variable
    >
    # NVIDIA CUDA Compiler (NVCC) Flags
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        -Xptxas -v           # Verbose PTX assembly output
        --use_fast_math      # Enable fast math optimizations
        -lineinfo            # Generate line info for profiling
        -DNDEBUG             # Disable debug assertions for release build
        --threads 4          # Parallelize NVCC compilation steps
    >
)
