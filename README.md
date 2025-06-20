# Team 58 - Prime Implicants for Quine-McCluskey

## Build

This project uses `cmake` as its build system.

- To generate the build files, run `cmake .`.
- To build the executable run `make -j8`. Substitute `8`with another number to adjust the number of parallel jobs.
- To run the executable, run `./prime_implicants`.

There are several options that can be passed to `cmake`:

- Add `-D LOG_LEVEL={0,1,2,3}` to set the log level:
  - `-D LOG_LEVEL=0` to log only errors
  - `-D LOG_LEVEL=1` to log only errors and warnings
  - `-D LOG_LEVEL=2` to log errors, warnings, and info messages (this is the default log level)
  - `-D LOG_LEVEL=3` to log errors, warnings, info messages, and debug messages
- Add `-D SANITIZE=ON` to enable address sanitizer and disable compiler optimizations to find memory-related errors. This will switch the compiler to `clang`, enable debug symbols and the `-fsanitize=address` flag.
Make sure to disable this option with `-D SANITIZE=OFF` before doing performance measurements.
- Use the `-D GENERATE_ASM=ON` option to enable generating assembly files in asm/ directory.

## Usage

- `./prime_implicants test <input_file1> [...]`: Test the program on the given test files. Also, run all tests relating to other functionality. Some small test files are given within the `test/` directory. For certain changes, it might be necessary to generate a test file with large `n` and `density` values via the `gentest` command.
- `./prime_implicants test_single <implementation> <input_file1> [...]`: Test a single implementation the given test files. This is useful for debugging a single implementation.
- `./prime_implicants measure <implementation> <n>`: Measure the performance of a single implementation on a random input of `n` variables. Its output will be appended to `measurements.csv`.
- `./prime_implicants measure_merge <merge_implementation> <n>`: Measure the performance of a single merge function implementation on a random input of `n` variables. Its output will be appended to `measurements_merge.csv`. Its output is rather noisy, so we did not include it in the report.
- `./prime_implicants help`: Print usage information.
- `./prime_implicants implementations`: Print a list of all available implementations.
- `./prime_implicants merge_implementations`: Print a list of all available merge function implementations.
- `./prime_implicants gentest <n> <density>`: Generate a test file with `n` variables and `density` percent density.

## Generating Traversals

For all implementations called `*load*`, traversal files are required.
The repository contains traversal files for the regular and DFS merge orders for up to `n=20`.
If you want to test `*load*` implementations with `n>20`, you need to generate traversals.
Run the following (you can adjust the number of bits if you like):

```sh
python generate_path.py --mode flat --out-dir traversals/flat/ 22
python generate_path.py --mode dfs --out-dir traversals/dfs/ 22
```

## Development

Before pushing changes, format code with `clang-format`.

## Known Issues

When measuring, the application uses `perf_event_open()` to measure cache misses and accesses.
If you encounter an error like `error in perf_event_open(): Permission denied`, you can try the following in order:

1. Run `sudo sysctl kernel.perf_event_paranoid=1` to allow the user to use `perf_event_open()` without root privileges.
1. Set the `CAP_PERFMON` capability for the user running the program. This can be done by running `sudo setcap cap_sys_admin+ep ./prime_implicants`.
1. Set the `CAP_SYS_ADMIN` capability for the user running the program. This can be done by running `sudo setcap cap_sys_admin+ep ./prime_implicants`.
1. Run the program as root.

### pre-commit hook

If you want, you can install the pre-commit hook, that makes sure the source code is formatted before pushing.

1. Make sure, you have installed the clang-format dependency.

2. Put the `scripts/pre-commit` file in `.git/hooks/` and mark it as executable

   ```bash
   cp scripts/pre-commit .git/hooks/
   chmod +x .git/hooks/pre-commit
   ```
