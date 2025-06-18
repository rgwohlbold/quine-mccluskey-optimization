cmake_minimum_required(VERSION 3.10)

project(prime_implicants C CXX)

file(GLOB SRC_FILES src/*.c src/implementations/*.c)
set(CPP_FILES src/implementations/hellman.cpp)


# Set compiler flags early so they're used everywhere
set(COMPILER_FLAGS -march=native)

add_executable(prime_implicants ${SRC_FILES} ${CPP_FILES})
# set_property(SOURCE ${CPP_FILES} PROPERTY LANGUAGE CXX)
# set_target_properties(prime_implicants PROPERTIES LINKER_LANGUAGE CXX)

# Generate annotated assembly files for all sources
set(ASM_OUTPUT_DIR "${CMAKE_BINARY_DIR}/asm")
file(MAKE_DIRECTORY ${ASM_OUTPUT_DIR})

foreach(src_file IN LISTS SRC_FILES CPP_FILES)
    get_filename_component(src_name ${src_file} NAME_WE)
    get_filename_component(src_ext ${src_file} EXT)
    if(src_ext STREQUAL ".c")
        add_custom_command(
            OUTPUT ${ASM_OUTPUT_DIR}/${src_name}.s
            COMMAND ${CMAKE_C_COMPILER} ${COMPILER_FLAGS} -fverbose-asm -S -g -o ${ASM_OUTPUT_DIR}/${src_name}.s ${src_file}
            DEPENDS ${src_file}
            COMMENT "Generating annotated assembly for ${src_file}"
            VERBATIM
        )
    elseif(src_ext STREQUAL ".cpp" OR src_ext STREQUAL ".cxx" OR src_ext STREQUAL ".cc")
        add_custom_command(
            OUTPUT ${ASM_OUTPUT_DIR}/${src_name}.s
            COMMAND ${CMAKE_CXX_COMPILER} ${COMPILER_FLAGS} -fverbose-asm -S -g -o ${ASM_OUTPUT_DIR}/${src_name}.s ${src_file}
            DEPENDS ${src_file}
            COMMENT "Generating annotated assembly for ${src_file}"
            VERBATIM
        )
    endif()
    list(APPEND ASM_OUTPUTS ${ASM_OUTPUT_DIR}/${src_name}.s)
endforeach()

add_custom_target(assembly ALL DEPENDS ${ASM_OUTPUTS})


option(SANITIZE "use sanitizers for debugging")
message(STATUS "Using SANITIZE=${SANITIZE}")

option(PROFILING "use profiling flags for compilation")
message(STATUS "Using PROFILING=${PROFILING}")

# Only clang supports AddressSanitizer
if(SANITIZE STREQUAL ON)
    set(CMAKE_C_COMPILER "/usr/bin/clang")
    set(COMPILER_FLAGS ${COMPILER_FLAGS} -fsanitize=address -fno-omit-frame-pointer -g)
    target_link_options(prime_implicants PUBLIC -fsanitize=address)
elseif(CMAKE_C_COMPILER STREQUAL "/usr/bin/clang")
    set(COMPILER_FLAGS ${COMPILER_FLAGS} -O3)
else()
    set(COMPILER_FLAGS ${COMPILER_FLAGS} -Ofast)
endif()

if(PROFILING STREQUAL ON)
    set(COMPILER_FLAGS ${COMPILER_FLAGS} -g -pg)
    target_link_options(prime_implicants PUBLIC -pg)
    add_definitions(-DVTUNE_API)

    # Set VTUNE_HOME
    find_path(VTUNE_INCLUDE_DIR ittnotify.h HINTS $ENV{VTUNE_HOME}/include)
    find_library(VTUNE_ITT_LIB ittnotify HINTS $ENV{VTUNE_HOME}/lib64)
    find_library(VTUNE_COLLECTOR_LIB ittnotify_collector HINTS $ENV{VTUNE_HOME}/lib64/runtime)

    if(NOT VTUNE_INCLUDE_DIR OR NOT VTUNE_ITT_LIB OR NOT VTUNE_COLLECTOR_LIB)
        message(FATAL_ERROR "Could not find ittnotify (VTune) headers or library. Please add VTUNE_HOME environment variable pointing to the VTune installation directory.")
    endif()

    include_directories(${VTUNE_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${VTUNE_ITT_LIB} ${VTUNE_COLLECTOR_LIB})
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${VTUNE_ITT_LIB}:${VTUNE_COLLECTOR_LIB}")
endif()

# Always add -mbmi2 to compile options for the target
target_compile_options(prime_implicants PUBLIC -Wall -Wextra ${COMPILER_FLAGS})

# Apply std flags per language
target_compile_options(prime_implicants PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-std=c11>
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++2a>
    ${COMPILER_FLAGS}
)

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
