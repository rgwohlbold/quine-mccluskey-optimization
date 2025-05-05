cmake_minimum_required(VERSION 3.10)

project(prime_implicants C)

file(GLOB SRC_FILES src/*.c src/implementations/*.c)
add_executable(prime_implicants ${SRC_FILES})

target_compile_options(prime_implicants PUBLIC -Wall -Wextra -std=c11 -march=native)

option(SANITIZE "use sanitizers for debugging")
message(STATUS "Using SANITIZE=${SANITIZE}")

if(SANITIZE STREQUAL ON)
    set(CMAKE_C_COMPILER "/usr/bin/clang")
    target_compile_options(prime_implicants PUBLIC -fsanitize=address -fno-omit-frame-pointer)
    target_link_options(prime_implicants PUBLIC -fsanitize=address)
else()
    target_compile_options(prime_implicants PUBLIC -Ofast)
endif()

### LOGGING OPTIONS BEGIN ###
option(LOG_NOCOLOR "Disable colored logs" OFF)
message(STATUS "Using LOG_LEVEL=${LOG_LEVEL}, NOCOLOR=${LOG_NOCOLOR}")
if(NOT DEFINED LOG_LEVEL)
set(LOG_LEVEL 1)
endif()

target_compile_definitions(prime_implicants PUBLIC LOG_LEVEL=${LOG_LEVEL})
if(LOG_NOCOLOR)
message(STATUS "LOG_NOCOLOR is enabled.")
target_compile_definitions(prime_implicants PUBLIC LOG_NOCOLOR)
endif()
### LOGGING OPTIONS END ###