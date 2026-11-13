# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.20)
project(gdn_recoverssm_ac LANGUAGES CXX)

# This is a Torch operator library, loaded through torch.ops.load_library. It
# does not need a Python module initializer or a separate installed OPP package.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_LINK_DEPENDS_USE_LINKER FALSE)
find_package(Torch REQUIRED)

foreach(required GDN_CANN_ROOT GDN_TORCH_NPU_ROOT GDN_OUTPUT_DIR)
  if(NOT DEFINED ${required})
    message(FATAL_ERROR "Missing ${required}: invoke the plugin startup builder")
  endif()
endforeach()

set(cann_arch "${GDN_CANN_ROOT}/${CMAKE_SYSTEM_PROCESSOR}-linux")
set(cann_includes
  "${GDN_CANN_ROOT}/include"
  "${GDN_CANN_ROOT}/include/op_common"
  "${GDN_CANN_ROOT}/pkg_inc"
  "${GDN_CANN_ROOT}/pkg_inc/op_common"
  "${cann_arch}/include"
  "${cann_arch}/include/op_common"
  "${cann_arch}/pkg_inc"
  "${cann_arch}/pkg_inc/op_common"
  "${GDN_CANN_ROOT}/compiler/tikcpp/include"
  "${GDN_CANN_ROOT}/compiler/tikcpp/tikcfw"
  "${GDN_CANN_ROOT}/compiler/tikcpp/tikcfw/interface"
  "${GDN_CANN_ROOT}/compiler/tikcpp/tikcfw/impl"
  "${GDN_CANN_ROOT}/compiler/ascendc/include/basic_api/interface"
  "${GDN_CANN_ROOT}/compiler/ascendc/include/basic_api/impl"
  "${GDN_CANN_ROOT}/compiler/ascendc/include/highlevel_api/impl"
  "${GDN_CANN_ROOT}/compiler/ascendc/include/highlevel_api/tiling"
  "${GDN_CANN_ROOT}/compiler/ascendc/impl/aicore/basic_api")
find_path(gdn_kernel_header kernel_operator.h HINTS ${cann_includes} NO_DEFAULT_PATH)
if(NOT gdn_kernel_header)
  message(FATAL_ERROR "kernel_operator.h is missing from the CANN development toolkit")
endif()

add_library(gdn_recoverssm_ac SHARED ops.cpp)
# BiSheng's AscendC mixed host/device mode, as used by CANN 9.1 fast launch.
# Do not replace with CUDA flags or enable fast-math: state arithmetic matters.
set_source_files_properties(ops.cpp PROPERTIES COMPILE_OPTIONS "--npu-arch=dav-2201;-xasc")
separate_arguments(torch_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(gdn_recoverssm_ac PRIVATE ${torch_flags} -O3)
target_include_directories(gdn_recoverssm_ac PRIVATE
  ${TORCH_INCLUDE_DIRS} "${GDN_TORCH_NPU_ROOT}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}" ${cann_includes} "${gdn_kernel_header}")
target_link_directories(gdn_recoverssm_ac PRIVATE
  "${GDN_TORCH_NPU_ROOT}/lib" "${GDN_CANN_ROOT}/lib64" "${cann_arch}/lib64")
target_link_libraries(gdn_recoverssm_ac PRIVATE
  ${TORCH_LIBRARIES} torch_npu ascendcl runtime platform)
set_target_properties(gdn_recoverssm_ac PROPERTIES
  PREFIX "lib" SUFFIX ".so" OUTPUT_NAME "gdn_recoverssm_ac"
  LIBRARY_OUTPUT_DIRECTORY "${GDN_OUTPUT_DIR}"
  BUILD_RPATH "${GDN_TORCH_NPU_ROOT}/lib;${TORCH_INSTALL_PREFIX}/lib;${GDN_CANN_ROOT}/lib64;${cann_arch}/lib64")
