cmake_minimum_required(VERSION 3.20)
project(artificial_peripheral_thalamus VERSION 1.0.0 LANGUAGES C CXX)

# ---------------------------------------------------------------------------------------------
# Artificial Peripheral Thalamus (APT) - device suite for the Cybernetic Neuromorphic Framework.
#   aptcore   C++20 library: evidence base, Module E/M/O physics, DSP, neuromorphic encoder,
#             bit-exact golden model of the FPGA datapath, host link, HIL plant
#   aptctl    command-line driver (spec, schedule, hil, codegen, vectors, decode)
#   aptfw     C99 firmware for the supervisory MCU, host-testable against a register-bus mock
# ---------------------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion)
  if(MINGW)
    # static runtime: Git-for-Windows ships an older libstdc++ that shadows the toolchain's
    add_link_options(-static)
  endif()
elseif(MSVC)
  add_compile_options(/W4 /permissive-)
endif()

add_library(aptcore
  src/evidence.cpp
  src/io.cpp
  src/afferent.cpp
  src/stability.cpp
  src/recovery.cpp
  src/schedule.cpp
)
target_include_directories(aptcore PUBLIC include)
target_compile_definitions(aptcore PRIVATE APT_SOURCE_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

# --- tests ---------------------------------------------------------------------------------
enable_testing()
foreach(t physics recovery schedule)
  add_executable(test_${t} tests/test_${t}.cpp)
  target_link_libraries(test_${t} PRIVATE aptcore)
  add_test(NAME ${t} COMMAND test_${t})
endforeach()
set_tests_properties(schedule PROPERTIES TIMEOUT 600)

