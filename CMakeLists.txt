cmake_minimum_required(VERSION 3.10)

project(prime_implicants C)

add_executable(prime_implicants main.c)

### LOGGING OPTIONS BEGIN ###
option(LOG_NOCOLOR "Disable colored logs" OFF)
message(STATUS "Using LOG_LEVEL= ${LOG_LEVEL}, NOCOLOR=${LOG_NOCOLOR}")
if(NOT DEFINED LOG_LEVEL)
set(LOG_LEVEL 1)
endif()

target_compile_definitions(prime_implicants PUBLIC LOG_LEVEL=${LOG_LEVEL})
if(LOG_NOCOLOR)
message(STATUS "LOG_NOCOLOR is enabled.")
target_compile_definitions(prime_implicants PUBLIC LOG_NOCOLOR)
endif()
### LOGGING OPTIONS END ###