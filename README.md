# KoalaBear

[![CI](https://img.shields.io/github/actions/workflow/status/libernet-xyz/koalabear/ci.yml?label=CI)](https://github.com/libernet-xyz/koalabear/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/starkom-koalabear)](https://crates.io/crates/starkom-koalabear)
[![license](https://img.shields.io/crates/l/starkom-koalabear)](https://github.com/libernet-xyz/koalabear/blob/main/LICENSE)

## Overview

Starkom's implementation of the KoalaBear field.

The order of the field is the prime $p = 2^{31} - 2^{24} + 1$, or `0x7F000001`.

This crate provides not only the base KoalaBear field but also the extension fields KoalaBear^2,
KoalaBear^4, and KoalaBear^8.

Most operations in this crate are accelerated with SIMD when compiling to **x86-64** or
**WebAssembly**, but they rely on plain Rust code for all other compilation targets. Furthermore,
SIMD support is not always present in WebAssembly and it's disabled by default in the Rust
toolchain, so in order to generate a SIMD-optimized WebAssembly binary you need to specify
`-C target-feature=+simd128` in `RUSTFLAGS` when compiling your module. Example:

```sh
$ RUSTFLAGS="-C target-feature=+simd128" cargo build --release --target wasm32-unknown-unknown
```

WebAssembly modules themselves don't have a way to check for SIMD support unfortunately, so you'll
typically need to ship two modules (one compiled with SIMD and one without) and your JavaScript code
will need to decide which one to instantiate by validating the SIMD one with
[`WebAssembly.validate()`][wasm-validate] or similar API.

[wasm-validate]: https://developer.mozilla.org/en-US/docs/WebAssembly/Reference/JavaScript_interface/validate_static
