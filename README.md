# KoalaBear

[![CI](https://img.shields.io/github/actions/workflow/status/libernet-xyz/koalabear/ci.yml?label=CI)](https://github.com/libernet-xyz/koalabear/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/starkom-koalabear)](https://crates.io/crates/starkom-koalabear)
[![license](https://img.shields.io/crates/l/starkom-koalabear)](https://github.com/libernet-xyz/koalabear/blob/main/LICENSE)

## Overview

Starkom's implementation of the KoalaBear field.

The order of the field is the prime $p = 2^{31} - 2^{24} + 1$, or `0x7F000001`.

This crate provides not only the base KoalaBear field but also the extension fields KoalaBear^2,
KoalaBear^4, and KoalaBear^8.
