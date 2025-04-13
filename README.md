# Team 58 - Prime Implicants for Quine-McCluskey

## Build

This project uses `cmake` as its build system.

- To generate the build files, run `cmake .`.
  Optionally, add `-D LOG_LEVEL={0,1,2}` to set the log level. Bigger number is more verbose.
  Also, add `-D SANITIZE=ON` to enable address sanitizer and disable compiler optimizations to find memory-related errors

- To build the executable run `make`.
- To run the executable, run `./prime_implicants`.

## Development

Before pushing changes, format code with `clang-format`.

### pre-commit hook

If you want, you can install the pre-commit hook, that makes sure the source code is formatted before pushing.

1. Make sure, you have installed the clang-format dependency.

2. Put the `scripts/pre-commit` file in `.git/hooks/` and mark it as executable

   ```bash
   cp scripts/pre-commit .git/hooks/
   chmod +x .git/hooks/pre-commit
   ```

## Known bugs

- Test case `all_minterms` fails for `prime_implicants_sparse`