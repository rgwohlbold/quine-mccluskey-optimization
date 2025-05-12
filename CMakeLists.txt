cmake_minimum_required(VERSION 3.10)

project(prime_implicants C)

file(GLOB SRC_FILES src/*.c src/implementations/*.c)
add_executable(prime_implicants ${SRC_FILES})


option(SANITIZE "use sanitizers for debugging")
message(STATUS "Using SANITIZE=${SANITIZE}")

set(COMPILER_FLAGS -march=native)

# Only clang supports AddressSanitizer
if(SANITIZE STREQUAL ON)
    set(CMAKE_C_COMPILER "/usr/bin/clang")
    set(COMPILER_FLAGS ${COMPILER_FLAGS} -fsanitize=address -fno-omit-frame-pointer)
    target_link_options(prime_implicants PUBLIC -fsanitize=address)
else()
    # Clang does not support -Ofast
    if(CMAKE_C_COMPILER STREQUAL "/usr/bin/clang")
        set(COMPILER_FLAGS ${COMPILER_FLAGS} -O3)
    else()
        set(COMPILER_FLAGS ${COMPILER_FLAGS} -Ofast)
    endif()
endif()

target_compile_options(prime_implicants PUBLIC -Wall -Wextra -std=c11 ${COMPILER_FLAGS})

# include information in executable
# get compiler version by calling compiler
execute_process(
  COMMAND ${CMAKE_C_COMPILER} --version
  OUTPUT_VARIABLE COMPILER_VERSION_OUTPUT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REPLACE "\n" ";" COMPILER_VERSION_LINES "${COMPILER_VERSION_OUTPUT}")
list(GET COMPILER_VERSION_LINES 0 COMPILER_VERSION_STR)

# join compiler flags list into string
string(JOIN " " COMPILER_FLAGS_STR ${COMPILER_FLAGS})

# get cpu by reading /proc/cpuinfo
if(EXISTS "/proc/cpuinfo")
    file(READ "/proc/cpuinfo" CPUINFO_CONTENTS)
    string(REGEX MATCH "model name[^\n]*" CPUINFO_LINE "${CPUINFO_CONTENTS}")
    string(REGEX REPLACE "model name[ \t]*:[ \t]*" "" CPU_MODEL_STR "${CPUINFO_LINE}")
elseif(APPLE)
  execute_process(
    COMMAND sysctl -n machdep.cpu.brand_string
    OUTPUT_VARIABLE CPU_MODEL_STR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
else()
    message("warning: could not detect cpu as /proc/cpuinfo does not exist")
endif()

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/system.c.in
    ${CMAKE_CURRENT_SOURCE_DIR}/src/system.c
)

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
