// Copyright 2026 The Libernet Team
// SPDX-License-Identifier: Apache-2.0

#![doc = include_str!("../README.md")]

mod helpers;

pub mod base;
pub mod kb2;
pub mod kb4;
pub mod kb8;

pub use base::Scalar as KB;
pub use kb2::Scalar as KB2;
pub use kb4::Scalar as KB4;
pub use kb8::Scalar as KB8;

/// The order of the KoalaBear field, `0x7F000001`.
pub const MODULUS: u32 = helpers::MODULUS;

/// The quadratic non-residue used to build the extension: `X^2 = QUADRATIC_NON_RESIDUE` in the base
/// field.
pub const QUADRATIC_NON_RESIDUE: u32 = helpers::QUADRATIC_NON_RESIDUE;

/// Alias for [`KB::from_const`].
#[inline(always)]
pub const fn from_const(value: u32) -> KB {
    KB::from_const(value)
}
