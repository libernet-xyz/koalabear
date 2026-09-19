use crate::base;
use crate::helpers::{
    CHARACTERS_LOWER_CASE, CHARACTERS_UPPER_CASE, MODULUS, QUADRATIC_NON_RESIDUE, kb_add, kb_add8,
    kb_from_montgomery, kb_mul2, kb_mul4, kb_mul8, kb_mul8x1, kb_sub, kb_sub8, kb_to_montgomery,
};
use crate::kb2;
use crate::kb4;
use anyhow::anyhow;
use primitive_types::{H512, U256, U512};
use rand_core::{CryptoRng, TryCryptoRng};
use starkom_ff::{Field, Field256};
use std::cmp::Ordering;
use std::fmt::{Binary, Debug, Display, Formatter, LowerHex, Octal, UpperHex};
use std::iter::{Product, Sum};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use std::str::FromStr;
use subtle::{
    Choice, ConditionallySelectable, ConstantTimeEq, ConstantTimeGreater, ConstantTimeLess,
    CtOption,
};

/// KoalaBear^8 extension field.
///
/// This is the degree-2 extension `KB4[Z] / (Z^2 - Y)` of [`kb4::Scalar`], where `Y` is the KB4
/// generator (`Y^2 = X` in KB2 and `X^2 = 3` in the base field). Equivalently, this is the
/// degree-8 extension `GF(p)[Z] / (Z^8 - 3)` of the KoalaBear field, where `p` is [`MODULUS`]. A
/// scalar `Scalar(a, b, c, d, e, f, g, h)` represents `KB4(a, b, c, d) * Z + KB4(e, f, g, h)`.
///
/// For all purposes other than field arithmetic (ordering, formatting, parsing, exponentiation,
/// etc.) a scalar is instead treated as the numeric value `a * MODULUS^7 + b * MODULUS^6 + ... +
/// g * MODULUS + h`. This gives every scalar a canonical representative in `0..(MODULUS^8)` for
/// those purposes.
///
/// NOTE: The `u32` words are stored in big-endian order: `Scalar::0` is the most significant and
/// `Scalar::7` is the least significant. All coefficients are in Montgomery form, exactly like the
/// inner value of a [`base::Scalar`].
#[derive(Default, Copy, Clone, PartialEq, Eq)]
pub struct Scalar(
    pub(crate) u32,
    pub(crate) u32,
    pub(crate) u32,
    pub(crate) u32,
    pub(crate) u32,
    pub(crate) u32,
    pub(crate) u32,
    pub(crate) u32,
);

impl Scalar {
    /// Constructs a KoalaBear^8 scalar from the raw values of its coefficients, ie.
    /// `KB4(a, b, c, d) * Z + KB4(e, f, g, h)`.
    ///
    /// Panics if any coefficient exceeds [`MODULUS`].
    #[inline]
    pub const fn from_coefficients(
        a: u32,
        b: u32,
        c: u32,
        d: u32,
        e: u32,
        f: u32,
        g: u32,
        h: u32,
    ) -> Self {
        Self(
            base::Scalar::from_const(a).0,
            base::Scalar::from_const(b).0,
            base::Scalar::from_const(c).0,
            base::Scalar::from_const(d).0,
            base::Scalar::from_const(e).0,
            base::Scalar::from_const(f).0,
            base::Scalar::from_const(g).0,
            base::Scalar::from_const(h).0,
        )
    }

    #[inline]
    const fn from_raw(value: u64) -> Self {
        const MODULUS_64: u64 = MODULUS as u64;
        Self::from_coefficients(
            0,
            0,
            0,
            0,
            0,
            (value / (MODULUS_64 * MODULUS_64)) as u32,
            ((value / MODULUS_64) % MODULUS_64) as u32,
            (value % MODULUS_64) as u32,
        )
    }

    /// Returns the raw (non-Montgomery) values of the coefficients.
    #[inline]
    const fn to_raw(&self) -> [u32; 8] {
        [
            kb_from_montgomery(self.0),
            kb_from_montgomery(self.1),
            kb_from_montgomery(self.2),
            kb_from_montgomery(self.3),
            kb_from_montgomery(self.4),
            kb_from_montgomery(self.5),
            kb_from_montgomery(self.6),
            kb_from_montgomery(self.7),
        ]
    }

    /// Constructs a KoalaBear^8 scalar from a raw 64-bit numeric value.
    #[inline]
    pub const fn from_const(value: u64) -> Self {
        Self::from_raw(value)
    }

    /// Returns the Montgomery values of the coefficients, in the layout of [`kb_mul8`].
    #[inline]
    const fn to_array(&self) -> [u32; 8] {
        [
            self.0, self.1, self.2, self.3, self.4, self.5, self.6, self.7,
        ]
    }

    /// Constructs a scalar from its numeric value, flagging it as invalid if the value is not lower
    /// than `MODULUS^8`.
    fn try_from_value(value: U256) -> CtOption<Self> {
        let modulus = U256::from(MODULUS);
        let mut remaining = value;
        let mut coefficients = [0u32; 8];
        for coefficient in coefficients.iter_mut().rev() {
            *coefficient = kb_to_montgomery((remaining % modulus).as_u32());
            remaining /= modulus;
        }
        let [a, b, c, d, e, f, g, h] = coefficients;
        CtOption::new(
            Self(a, b, c, d, e, f, g, h),
            ((remaining == U256::zero()) as u8).into(),
        )
    }
}

impl Ord for Scalar {
    fn cmp(&self, other: &Self) -> Ordering {
        self.to_raw().cmp(&other.to_raw())
    }
}

impl PartialOrd for Scalar {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl ConstantTimeEq for Scalar {
    fn ct_eq(&self, other: &Self) -> Choice {
        ((self.0 == other.0) as u8
            & (self.1 == other.1) as u8
            & (self.2 == other.2) as u8
            & (self.3 == other.3) as u8
            & (self.4 == other.4) as u8
            & (self.5 == other.5) as u8
            & (self.6 == other.6) as u8
            & (self.7 == other.7) as u8)
            .into()
    }
}

impl ConstantTimeGreater for Scalar {
    fn ct_gt(&self, other: &Self) -> Choice {
        ((self.to_raw() > other.to_raw()) as u8).into()
    }
}

impl ConstantTimeLess for Scalar {}

impl ConditionallySelectable for Scalar {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        if choice.into() { *b } else { *a }
    }
}

impl Add<Self> for Scalar {
    type Output = Scalar;

    fn add(self, rhs: Self) -> Self::Output {
        let [c0, c1, c2, c3, c4, c5, c6, c7] = kb_add8(self.to_array(), rhs.to_array());
        Self(c0, c1, c2, c3, c4, c5, c6, c7)
    }
}

impl<'a> Add<&'a Self> for Scalar {
    type Output = Scalar;

    fn add(self, rhs: &'a Self) -> Self::Output {
        self.add(*rhs)
    }
}

impl AddAssign<Self> for Scalar {
    fn add_assign(&mut self, rhs: Self) {
        *self = self.add(rhs);
    }
}

impl<'a> AddAssign<&'a Self> for Scalar {
    fn add_assign(&mut self, rhs: &'a Self) {
        self.add_assign(*rhs);
    }
}

impl Add<base::Scalar> for Scalar {
    type Output = Scalar;

    fn add(self, rhs: base::Scalar) -> Self::Output {
        Self(
            self.0,
            self.1,
            self.2,
            self.3,
            self.4,
            self.5,
            self.6,
            kb_add(self.7, rhs.0),
        )
    }
}

impl<'a> Add<&'a base::Scalar> for Scalar {
    type Output = Scalar;

    fn add(self, rhs: &'a base::Scalar) -> Self::Output {
        self.add(*rhs)
    }
}

impl AddAssign<base::Scalar> for Scalar {
    fn add_assign(&mut self, rhs: base::Scalar) {
        self.7 = kb_add(self.7, rhs.0);
    }
}

impl<'a> AddAssign<&'a base::Scalar> for Scalar {
    fn add_assign(&mut self, rhs: &'a base::Scalar) {
        self.add_assign(*rhs);
    }
}

impl Add<kb2::Scalar> for Scalar {
    type Output = Scalar;

    fn add(self, rhs: kb2::Scalar) -> Self::Output {
        Self(
            self.0,
            self.1,
            self.2,
            self.3,
            self.4,
            self.5,
            kb_add(self.6, rhs.0),
            kb_add(self.7, rhs.1),
        )
    }
}

impl<'a> Add<&'a kb2::Scalar> for Scalar {
    type Output = Scalar;

    fn add(self, rhs: &'a kb2::Scalar) -> Self::Output {
        self.add(*rhs)
    }
}

impl AddAssign<kb2::Scalar> for Scalar {
    fn add_assign(&mut self, rhs: kb2::Scalar) {
        self.6 = kb_add(self.6, rhs.0);
        self.7 = kb_add(self.7, rhs.1);
    }
}

impl<'a> AddAssign<&'a kb2::Scalar> for Scalar {
    fn add_assign(&mut self, rhs: &'a kb2::Scalar) {
        self.add_assign(*rhs);
    }
}

impl Add<kb4::Scalar> for Scalar {
    type Output = Scalar;

    fn add(self, rhs: kb4::Scalar) -> Self::Output {
        Self(
            self.0,
            self.1,
            self.2,
            self.3,
            kb_add(self.4, rhs.0),
            kb_add(self.5, rhs.1),
            kb_add(self.6, rhs.2),
            kb_add(self.7, rhs.3),
        )
    }
}

impl<'a> Add<&'a kb4::Scalar> for Scalar {
    type Output = Scalar;

    fn add(self, rhs: &'a kb4::Scalar) -> Self::Output {
        self.add(*rhs)
    }
}

impl AddAssign<kb4::Scalar> for Scalar {
    fn add_assign(&mut self, rhs: kb4::Scalar) {
        self.4 = kb_add(self.4, rhs.0);
        self.5 = kb_add(self.5, rhs.1);
        self.6 = kb_add(self.6, rhs.2);
        self.7 = kb_add(self.7, rhs.3);
    }
}

impl<'a> AddAssign<&'a kb4::Scalar> for Scalar {
    fn add_assign(&mut self, rhs: &'a kb4::Scalar) {
        self.add_assign(*rhs);
    }
}

impl Neg for Scalar {
    type Output = Scalar;

    fn neg(self) -> Self::Output {
        let [c0, c1, c2, c3, c4, c5, c6, c7] = kb_sub8([0; 8], self.to_array());
        Self(c0, c1, c2, c3, c4, c5, c6, c7)
    }
}

impl Sub<Self> for Scalar {
    type Output = Scalar;

    fn sub(self, rhs: Self) -> Self::Output {
        let [c0, c1, c2, c3, c4, c5, c6, c7] = kb_sub8(self.to_array(), rhs.to_array());
        Self(c0, c1, c2, c3, c4, c5, c6, c7)
    }
}

impl<'a> Sub<&'a Self> for Scalar {
    type Output = Scalar;

    fn sub(self, rhs: &'a Self) -> Self::Output {
        self.sub(*rhs)
    }
}

impl SubAssign<Self> for Scalar {
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.sub(rhs);
    }
}

impl<'a> SubAssign<&'a Self> for Scalar {
    fn sub_assign(&mut self, rhs: &'a Self) {
        self.sub_assign(*rhs);
    }
}

impl Sub<base::Scalar> for Scalar {
    type Output = Scalar;

    fn sub(self, rhs: base::Scalar) -> Self::Output {
        Self(
            self.0,
            self.1,
            self.2,
            self.3,
            self.4,
            self.5,
            self.6,
            kb_sub(self.7, rhs.0),
        )
    }
}

impl<'a> Sub<&'a base::Scalar> for Scalar {
    type Output = Scalar;

    fn sub(self, rhs: &'a base::Scalar) -> Self::Output {
        self.sub(*rhs)
    }
}

impl SubAssign<base::Scalar> for Scalar {
    fn sub_assign(&mut self, rhs: base::Scalar) {
        self.7 = kb_sub(self.7, rhs.0);
    }
}

impl<'a> SubAssign<&'a base::Scalar> for Scalar {
    fn sub_assign(&mut self, rhs: &'a base::Scalar) {
        self.sub_assign(*rhs);
    }
}

impl Sub<kb2::Scalar> for Scalar {
    type Output = Scalar;

    fn sub(self, rhs: kb2::Scalar) -> Self::Output {
        Self(
            self.0,
            self.1,
            self.2,
            self.3,
            self.4,
            self.5,
            kb_sub(self.6, rhs.0),
            kb_sub(self.7, rhs.1),
        )
    }
}

impl<'a> Sub<&'a kb2::Scalar> for Scalar {
    type Output = Scalar;

    fn sub(self, rhs: &'a kb2::Scalar) -> Self::Output {
        self.sub(*rhs)
    }
}

impl SubAssign<kb2::Scalar> for Scalar {
    fn sub_assign(&mut self, rhs: kb2::Scalar) {
        self.6 = kb_sub(self.6, rhs.0);
        self.7 = kb_sub(self.7, rhs.1);
    }
}

impl<'a> SubAssign<&'a kb2::Scalar> for Scalar {
    fn sub_assign(&mut self, rhs: &'a kb2::Scalar) {
        self.sub_assign(*rhs);
    }
}

impl Sub<kb4::Scalar> for Scalar {
    type Output = Scalar;

    fn sub(self, rhs: kb4::Scalar) -> Self::Output {
        Self(
            self.0,
            self.1,
            self.2,
            self.3,
            kb_sub(self.4, rhs.0),
            kb_sub(self.5, rhs.1),
            kb_sub(self.6, rhs.2),
            kb_sub(self.7, rhs.3),
        )
    }
}

impl<'a> Sub<&'a kb4::Scalar> for Scalar {
    type Output = Scalar;

    fn sub(self, rhs: &'a kb4::Scalar) -> Self::Output {
        self.sub(*rhs)
    }
}

impl SubAssign<kb4::Scalar> for Scalar {
    fn sub_assign(&mut self, rhs: kb4::Scalar) {
        self.4 = kb_sub(self.4, rhs.0);
        self.5 = kb_sub(self.5, rhs.1);
        self.6 = kb_sub(self.6, rhs.2);
        self.7 = kb_sub(self.7, rhs.3);
    }
}

impl<'a> SubAssign<&'a kb4::Scalar> for Scalar {
    fn sub_assign(&mut self, rhs: &'a kb4::Scalar) {
        self.sub_assign(*rhs);
    }
}

impl Mul<Self> for Scalar {
    type Output = Scalar;

    fn mul(self, rhs: Self) -> Self::Output {
        let [a, b, c, d, e, f, g, h] = kb_mul8(self.to_array(), rhs.to_array());
        Self(a, b, c, d, e, f, g, h)
    }
}

impl<'a> Mul<&'a Self> for Scalar {
    type Output = Scalar;

    fn mul(self, rhs: &'a Self) -> Self::Output {
        self.mul(*rhs)
    }
}

impl MulAssign<Self> for Scalar {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.mul(rhs);
    }
}

impl<'a> MulAssign<&'a Self> for Scalar {
    fn mul_assign(&mut self, rhs: &'a Self) {
        self.mul_assign(*rhs);
    }
}

impl Mul<base::Scalar> for Scalar {
    type Output = Scalar;

    fn mul(self, rhs: base::Scalar) -> Self::Output {
        let [c0, c1, c2, c3, c4, c5, c6, c7] = kb_mul8x1(self.to_array(), rhs.0);
        Self(c0, c1, c2, c3, c4, c5, c6, c7)
    }
}

impl<'a> Mul<&'a base::Scalar> for Scalar {
    type Output = Scalar;

    fn mul(self, rhs: &'a base::Scalar) -> Self::Output {
        self.mul(*rhs)
    }
}

impl MulAssign<base::Scalar> for Scalar {
    fn mul_assign(&mut self, rhs: base::Scalar) {
        *self = self.mul(rhs);
    }
}

impl<'a> MulAssign<&'a base::Scalar> for Scalar {
    fn mul_assign(&mut self, rhs: &'a base::Scalar) {
        self.mul_assign(*rhs);
    }
}

impl Mul<kb2::Scalar> for Scalar {
    type Output = Scalar;

    fn mul(self, rhs: kb2::Scalar) -> Self::Output {
        let [a, b] = kb_mul2([self.0, self.1], [rhs.0, rhs.1]);
        let [c, d] = kb_mul2([self.2, self.3], [rhs.0, rhs.1]);
        let [e, f] = kb_mul2([self.4, self.5], [rhs.0, rhs.1]);
        let [g, h] = kb_mul2([self.6, self.7], [rhs.0, rhs.1]);
        Self(a, b, c, d, e, f, g, h)
    }
}

impl<'a> Mul<&'a kb2::Scalar> for Scalar {
    type Output = Scalar;

    fn mul(self, rhs: &'a kb2::Scalar) -> Self::Output {
        self.mul(*rhs)
    }
}

impl MulAssign<kb2::Scalar> for Scalar {
    fn mul_assign(&mut self, rhs: kb2::Scalar) {
        *self = self.mul(rhs);
    }
}

impl<'a> MulAssign<&'a kb2::Scalar> for Scalar {
    fn mul_assign(&mut self, rhs: &'a kb2::Scalar) {
        self.mul_assign(*rhs);
    }
}

impl Mul<kb4::Scalar> for Scalar {
    type Output = Scalar;

    fn mul(self, rhs: kb4::Scalar) -> Self::Output {
        let [a, b, c, d] = kb_mul4(
            [self.0, self.1, self.2, self.3],
            [rhs.0, rhs.1, rhs.2, rhs.3],
        );
        let [e, f, g, h] = kb_mul4(
            [self.4, self.5, self.6, self.7],
            [rhs.0, rhs.1, rhs.2, rhs.3],
        );
        Self(a, b, c, d, e, f, g, h)
    }
}

impl<'a> Mul<&'a kb4::Scalar> for Scalar {
    type Output = Scalar;

    fn mul(self, rhs: &'a kb4::Scalar) -> Self::Output {
        self.mul(*rhs)
    }
}

impl MulAssign<kb4::Scalar> for Scalar {
    fn mul_assign(&mut self, rhs: kb4::Scalar) {
        *self = self.mul(rhs);
    }
}

impl<'a> MulAssign<&'a kb4::Scalar> for Scalar {
    fn mul_assign(&mut self, rhs: &'a kb4::Scalar) {
        self.mul_assign(*rhs);
    }
}

impl Div<Self> for Scalar {
    type Output = Scalar;

    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.invert_unwrap()
    }
}

impl<'a> Div<&'a Self> for Scalar {
    type Output = Scalar;

    fn div(self, rhs: &'a Self) -> Self::Output {
        self.div(*rhs)
    }
}

impl DivAssign<Self> for Scalar {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self * rhs.invert_unwrap();
    }
}

impl<'a> DivAssign<&'a Self> for Scalar {
    fn div_assign(&mut self, rhs: &'a Self) {
        self.div_assign(*rhs);
    }
}

impl Div<base::Scalar> for Scalar {
    type Output = Scalar;

    fn div(self, rhs: base::Scalar) -> Self::Output {
        self * rhs.invert_unwrap()
    }
}

impl<'a> Div<&'a base::Scalar> for Scalar {
    type Output = Scalar;

    fn div(self, rhs: &'a base::Scalar) -> Self::Output {
        self.div(*rhs)
    }
}

impl DivAssign<base::Scalar> for Scalar {
    fn div_assign(&mut self, rhs: base::Scalar) {
        *self = *self * rhs.invert_unwrap();
    }
}

impl<'a> DivAssign<&'a base::Scalar> for Scalar {
    fn div_assign(&mut self, rhs: &'a base::Scalar) {
        self.div_assign(*rhs);
    }
}

impl Div<kb2::Scalar> for Scalar {
    type Output = Scalar;

    fn div(self, rhs: kb2::Scalar) -> Self::Output {
        self * rhs.invert_unwrap()
    }
}

impl<'a> Div<&'a kb2::Scalar> for Scalar {
    type Output = Scalar;

    fn div(self, rhs: &'a kb2::Scalar) -> Self::Output {
        self.div(*rhs)
    }
}

impl DivAssign<kb2::Scalar> for Scalar {
    fn div_assign(&mut self, rhs: kb2::Scalar) {
        *self = *self * rhs.invert_unwrap();
    }
}

impl<'a> DivAssign<&'a kb2::Scalar> for Scalar {
    fn div_assign(&mut self, rhs: &'a kb2::Scalar) {
        self.div_assign(*rhs);
    }
}

impl Div<kb4::Scalar> for Scalar {
    type Output = Scalar;

    fn div(self, rhs: kb4::Scalar) -> Self::Output {
        self * rhs.invert_unwrap()
    }
}

impl<'a> Div<&'a kb4::Scalar> for Scalar {
    type Output = Scalar;

    fn div(self, rhs: &'a kb4::Scalar) -> Self::Output {
        self.div(*rhs)
    }
}

impl DivAssign<kb4::Scalar> for Scalar {
    fn div_assign(&mut self, rhs: kb4::Scalar) {
        *self = *self * rhs.invert_unwrap();
    }
}

impl<'a> DivAssign<&'a kb4::Scalar> for Scalar {
    fn div_assign(&mut self, rhs: &'a kb4::Scalar) {
        self.div_assign(*rhs);
    }
}

impl Sum<Scalar> for Scalar {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |a, b| a + b)
    }
}

impl<'a> Sum<&'a Scalar> for Scalar {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |a, b| a + b)
    }
}

impl Product<Scalar> for Scalar {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |a, b| a * b)
    }
}

impl<'a> Product<&'a Scalar> for Scalar {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |a, b| a * b)
    }
}

impl Debug for Scalar {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Scalar({:#066x})", self)
    }
}

impl Display for Scalar {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#066x}", self)
    }
}

impl Binary for Scalar {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let prefix = if f.alternate() { "0b" } else { "" };
        f.pad_integral(true, prefix, &self.to_str_radix(2, 0, false))
    }
}

impl Octal for Scalar {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let prefix = if f.alternate() { "0o" } else { "" };
        f.pad_integral(true, prefix, &self.to_str_radix(8, 0, false))
    }
}

impl LowerHex for Scalar {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let prefix = if f.alternate() { "0x" } else { "" };
        f.pad_integral(true, prefix, &self.to_str_radix(16, 0, false))
    }
}

impl UpperHex for Scalar {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let prefix = if f.alternate() { "0x" } else { "" };
        f.pad_integral(true, prefix, &self.to_str_radix(16, 0, true))
    }
}

impl FromStr for Scalar {
    type Err = std::fmt::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.starts_with("0x") || s.starts_with("0X") {
            Self::from_str_radix(&s[2..], 16)
        } else if s.starts_with("0b") || s.starts_with("0B") {
            Self::from_str_radix(&s[2..], 2)
        } else if s.starts_with("0o") || s.starts_with("0O") {
            Self::from_str_radix(&s[2..], 8)
        } else if s.starts_with("0") {
            Self::from_str_radix(s, 8)
        } else {
            Self::from_str_radix(s, 10)
        }
    }
}

impl From<u8> for Scalar {
    fn from(value: u8) -> Self {
        Self::from_coefficients(0, 0, 0, 0, 0, 0, 0, value as u32)
    }
}

impl From<u16> for Scalar {
    fn from(value: u16) -> Self {
        Self::from_coefficients(0, 0, 0, 0, 0, 0, 0, value as u32)
    }
}

impl From<u32> for Scalar {
    fn from(value: u32) -> Self {
        Self::from_raw(value as u64)
    }
}

impl From<u64> for Scalar {
    fn from(value: u64) -> Self {
        Self::from_raw(value)
    }
}

impl From<u128> for Scalar {
    fn from(value: u128) -> Self {
        const MODULUS_128: u128 = MODULUS as u128;
        let h = value % MODULUS_128;
        let value = value / MODULUS_128;
        let g = value % MODULUS_128;
        let value = value / MODULUS_128;
        let f = value % MODULUS_128;
        let value = value / MODULUS_128;
        let e = value % MODULUS_128;
        let d = value / MODULUS_128;
        Self::from_coefficients(0, 0, 0, d as u32, e as u32, f as u32, g as u32, h as u32)
    }
}

impl From<base::Scalar> for Scalar {
    fn from(value: base::Scalar) -> Self {
        Self(0, 0, 0, 0, 0, 0, 0, value.0)
    }
}

impl From<kb2::Scalar> for Scalar {
    fn from(value: kb2::Scalar) -> Self {
        Self(0, 0, 0, 0, 0, 0, value.0, value.1)
    }
}

impl From<kb4::Scalar> for Scalar {
    fn from(value: kb4::Scalar) -> Self {
        Self(0, 0, 0, 0, value.0, value.1, value.2, value.3)
    }
}

impl TryFrom<usize> for Scalar {
    type Error = anyhow::Error;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        Ok(Self::from_raw(value as u64))
    }
}

impl TryFrom<U256> for Scalar {
    type Error = anyhow::Error;

    fn try_from(value: U256) -> Result<Self, Self::Error> {
        Self::try_from_value(value)
            .into_option()
            .ok_or_else(|| anyhow!("{:#x} exceeds the KoalaBear^8 range", value))
    }
}

impl Field for Scalar {
    const MODULUS: &'static str =
        "0xf06e44682c2aa440f5f26a5ae174900568744cd653c806e41c0003f8000001";

    const CHARACTERISTIC: &'static str = "0x7f000001";

    const LEN: usize = 32;

    const ZERO: Self = Self(0, 0, 0, 0, 0, 0, 0, 0);

    const ONE: Self = Self::from_coefficients(0, 0, 0, 0, 0, 0, 0, 1);

    const MAX: Self = Self::from_coefficients(
        MODULUS - 1,
        MODULUS - 1,
        MODULUS - 1,
        MODULUS - 1,
        MODULUS - 1,
        MODULUS - 1,
        MODULUS - 1,
        MODULUS - 1,
    );

    const S: usize = 27;

    const MULTIPLICATIVE_GENERATOR: Self = Self::from_coefficients(
        0x3426065a, 0x13ac7c06, 0x023b1b71, 0x642192ae, 0x68087c79, 0x5481f83a, 0x5938d3c3,
        0x31ed2f75,
    );

    const MINUS_TWO: Self = Self::from_coefficients(
        MODULUS - 1,
        MODULUS - 1,
        MODULUS - 1,
        MODULUS - 1,
        MODULUS - 1,
        MODULUS - 1,
        MODULUS - 1,
        MODULUS - 2,
    );

    const TWO_INV: Self = Self::from_coefficients(0, 0, 0, 0, 0, 0, 0, 0x3f800001);

    const ROOT_OF_UNITY: Self = Self::from_coefficients(0x00daf26b, 0, 0, 0, 0, 0, 0, 0);

    const ROOT_OF_UNITY_INV: Self = Self::from_coefficients(0, 0, 0, 0x3bd2962a, 0, 0, 0, 0);

    const DELTA: Self = Self::from_coefficients(
        0x7d9ac33e, 0x10751fab, 0x683bbb45, 0x7c69f9f1, 0x354eae02, 0x49af8d25, 0x5db10006,
        0x2ce4a61f,
    );

    fn is_odd(&self) -> Choice {
        let [a, b, c, d, e, f, g, h] = self.to_raw();
        (((a ^ b ^ c ^ d ^ e ^ f ^ g ^ h) & 1) as u8).into()
    }

    fn try_random<R: TryCryptoRng>(rng: &mut R) -> Result<Self, R::Error> {
        Ok(Self(
            base::Scalar::try_random(rng)?.0,
            base::Scalar::try_random(rng)?.0,
            base::Scalar::try_random(rng)?.0,
            base::Scalar::try_random(rng)?.0,
            base::Scalar::try_random(rng)?.0,
            base::Scalar::try_random(rng)?.0,
            base::Scalar::try_random(rng)?.0,
            base::Scalar::try_random(rng)?.0,
        ))
    }

    fn random<R: CryptoRng>(rng: &mut R) -> Self {
        Self(
            base::Scalar::random(rng).0,
            base::Scalar::random(rng).0,
            base::Scalar::random(rng).0,
            base::Scalar::random(rng).0,
            base::Scalar::random(rng).0,
            base::Scalar::random(rng).0,
            base::Scalar::random(rng).0,
            base::Scalar::random(rng).0,
        )
    }

    fn random_default() -> Self {
        Self(
            base::Scalar::random_default().0,
            base::Scalar::random_default().0,
            base::Scalar::random_default().0,
            base::Scalar::random_default().0,
            base::Scalar::random_default().0,
            base::Scalar::random_default().0,
            base::Scalar::random_default().0,
            base::Scalar::random_default().0,
        )
    }

    fn invert(&self) -> CtOption<Self> {
        let a = kb4::Scalar(self.0, self.1, self.2, self.3);
        let b = kb4::Scalar(self.4, self.5, self.6, self.7);
        let a2 = a * a;
        let a2y = kb4::Scalar(
            a2.2,
            a2.3,
            a2.1,
            (base::Scalar(a2.0) * base::Scalar::from_const(QUADRATIC_NON_RESIDUE)).0,
        );
        let norm = b * b - a2y;
        let conjugate = Self(
            kb_sub(0, self.0),
            kb_sub(0, self.1),
            kb_sub(0, self.2),
            kb_sub(0, self.3),
            self.4,
            self.5,
            self.6,
            self.7,
        );
        norm.invert().map(|inverse_norm| conjugate * inverse_norm)
    }

    fn invert_vartime(&self) -> Option<Self> {
        let a = kb4::Scalar(self.0, self.1, self.2, self.3);
        let b = kb4::Scalar(self.4, self.5, self.6, self.7);
        let a2 = a * a;
        let a2y = kb4::Scalar(
            a2.2,
            a2.3,
            a2.1,
            (base::Scalar(a2.0) * base::Scalar::from_const(QUADRATIC_NON_RESIDUE)).0,
        );
        let norm = b * b - a2y;
        let conjugate = Self(
            kb_sub(0, self.0),
            kb_sub(0, self.1),
            kb_sub(0, self.2),
            kb_sub(0, self.3),
            self.4,
            self.5,
            self.6,
            self.7,
        );
        norm.invert_vartime()
            .map(|inverse_norm| conjugate * inverse_norm)
    }

    fn pow(mut self, exp: Self) -> Self {
        let mut exponent = exp.to_u256();
        let mut result = Self::ONE;
        for _ in 0..Self::NUM_BITS {
            let product = result * self;
            let bit = ((exponent & U256::one()).as_u64() as u8).into();
            result = Scalar::conditional_select(&result, &product, bit);
            exponent >>= 1;
            self = self.square();
        }
        result
    }

    fn pow_vartime(mut self, exp: Self) -> Self {
        let mut exponent = exp.to_u256();
        let mut result = Self::ONE;
        while exponent != U256::zero() {
            if (exponent & U256::one()) != U256::zero() {
                result *= self;
            }
            exponent >>= 1;
            self = self.square();
        }
        result
    }

    fn div_int(&self, rhs: &Self) -> (Self, Self) {
        let lhs_value = self.to_u256();
        let rhs_value = rhs.to_u256();
        (
            Self::try_from(lhs_value / rhs_value).unwrap(),
            Self::try_from(lhs_value % rhs_value).unwrap(),
        )
    }

    fn try_from_le_bytes(bytes: &[u8]) -> CtOption<Self> {
        Self::try_from_value(U256::from_little_endian(bytes))
    }

    fn try_from_be_bytes(bytes: &[u8]) -> CtOption<Self> {
        Self::try_from_value(U256::from_big_endian(bytes))
    }

    fn from_str_radix(s: &str, radix: usize) -> Result<Self, std::fmt::Error> {
        assert!(radix >= 2 && radix <= 36);
        if s.is_empty() {
            return Err(std::fmt::Error);
        }
        let mut value = U256::zero();
        let radix_u256 = U256::from(radix);
        for byte in s.bytes() {
            let digit = CHARACTERS_UPPER_CASE[..radix]
                .iter()
                .position(|&c| c == byte)
                .or_else(|| {
                    CHARACTERS_LOWER_CASE[..radix]
                        .iter()
                        .position(|&c| c == byte)
                })
                .ok_or(std::fmt::Error)?;
            value = value
                .checked_mul(radix_u256)
                .ok_or(std::fmt::Error)?
                .checked_add(U256::from(digit))
                .ok_or(std::fmt::Error)?;
        }
        Self::try_from(value).map_err(|_| std::fmt::Error)
    }

    fn to_str_radix(&self, radix: usize, pad_to: usize, upper_case: bool) -> String {
        assert!(radix >= 2 && radix <= 36);
        let characters = if upper_case {
            CHARACTERS_UPPER_CASE
        } else {
            CHARACTERS_LOWER_CASE
        };
        let mut value = self.to_u256();
        let mut s = String::default();
        let radix = U256::from(radix);
        while value != U256::zero() {
            let digit = value % radix;
            s.push(characters[digit.as_u64() as usize] as char);
            value /= radix;
        }
        if s.is_empty() {
            s.push('0');
        }
        while s.len() < pad_to {
            s.push('0');
        }
        s.chars().rev().collect()
    }

    fn try_to_u8(&self) -> Option<u8> {
        let value = self.to_u256();
        if value > U256::from(u8::MAX) {
            None
        } else {
            Some(value.low_u32() as u8)
        }
    }

    fn try_to_u16(&self) -> Option<u16> {
        let value = self.to_u256();
        if value > U256::from(u16::MAX) {
            None
        } else {
            Some(value.low_u32() as u16)
        }
    }

    fn to_u256(&self) -> U256 {
        let modulus = U256::from(MODULUS);
        self.to_raw()
            .iter()
            .fold(U256::zero(), |value, &coefficient| {
                value * modulus + U256::from(coefficient)
            })
    }

    fn to_u512(&self) -> U512 {
        self.to_u256().into()
    }
}

impl Field256 for Scalar {
    fn to_le_bytes(&self) -> [u8; 32] {
        self.to_u256().to_little_endian()
    }

    fn to_be_bytes(&self) -> [u8; 32] {
        self.to_u256().to_big_endian()
    }

    fn from_u512_mod_n(u512: U512) -> Self {
        let modulus = U512::from(MODULUS);
        let modulus_pow2 = modulus * modulus;
        let modulus_pow4 = modulus_pow2 * modulus_pow2;
        let modulus_pow8 = modulus_pow4 * modulus_pow4;
        let value = U256::try_from(u512 % modulus_pow8).unwrap();
        Self::try_from(value).unwrap()
    }

    fn from_h512(h512: H512) -> Self {
        Self::from_u512_mod_n(U512::from_little_endian(h512.as_bytes()))
    }

    fn try_to_u32(&self) -> CtOption<u32> {
        let value = self.to_u256();
        CtOption::new(
            value.low_u32(),
            ((value <= U256::from(u32::MAX)) as u8).into(),
        )
    }

    fn try_to_u64(&self) -> CtOption<u64> {
        let value = self.to_u256();
        CtOption::new(
            value.low_u64(),
            ((value <= U256::from(u64::MAX)) as u8).into(),
        )
    }

    fn try_to_u128(&self) -> CtOption<u128> {
        let value = self.to_u256();
        CtOption::new(
            value.low_u128(),
            ((value <= U256::from(u128::MAX)) as u8).into(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[inline]
    const fn from_const(value: u64) -> Scalar {
        Scalar::from_const(value)
    }

    #[inline]
    const fn from_coefficients(
        a: u32,
        b: u32,
        c: u32,
        d: u32,
        e: u32,
        f: u32,
        g: u32,
        h: u32,
    ) -> Scalar {
        Scalar::from_coefficients(a, b, c, d, e, f, g, h)
    }

    #[inline]
    fn parse_scalar(s: &'static str) -> Scalar {
        s.parse().unwrap()
    }

    /// `MODULUS^8`, ie. one past the largest representable value.
    fn modulus_pow8() -> U256 {
        let modulus = U256::from(MODULUS);
        let modulus_pow2 = modulus * modulus;
        let modulus_pow4 = modulus_pow2 * modulus_pow2;
        modulus_pow4 * modulus_pow4
    }

    #[test]
    fn test_from_const() {
        assert_eq!(from_const(0), Scalar::ZERO);
        assert_eq!(from_const(1), Scalar::ONE);
        assert_eq!(
            from_const(MODULUS as u64),
            from_coefficients(0, 0, 0, 0, 0, 0, 1, 0)
        );
        assert_eq!(
            from_const(MODULUS as u64 + 1),
            from_coefficients(0, 0, 0, 0, 0, 0, 1, 1)
        );
        assert_eq!(
            from_const((MODULUS as u64) * (MODULUS as u64)),
            from_coefficients(0, 0, 0, 0, 0, 1, 0, 0)
        );
        assert_eq!(
            from_const(u64::MAX),
            from_coefficients(0, 0, 0, 0, 0, 4, 134746136, 402124771)
        );
    }

    #[test]
    fn test_from_coefficients() {
        assert_eq!(from_coefficients(0, 0, 0, 0, 0, 0, 0, 0), Scalar::ZERO);
        assert_eq!(from_coefficients(0, 0, 0, 0, 0, 0, 0, 1), Scalar::ONE);
        assert_eq!(
            from_coefficients(0, 0, 0, 0, 0, 0, 1, 2),
            from_const(MODULUS as u64 + 2)
        );
    }

    #[test]
    #[should_panic(expected = "invalid KoalaBear value")]
    fn test_from_coefficients_out_of_range() {
        from_coefficients(MODULUS, 0, 0, 0, 0, 0, 0, 0);
    }

    #[test]
    fn test_modulus() {
        assert_eq!(
            Scalar::MODULUS,
            "0xf06e44682c2aa440f5f26a5ae174900568744cd653c806e41c0003f8000001"
        );
        assert_eq!(
            Scalar::MAX,
            from_coefficients(
                MODULUS - 1,
                MODULUS - 1,
                MODULUS - 1,
                MODULUS - 1,
                MODULUS - 1,
                MODULUS - 1,
                MODULUS - 1,
                MODULUS - 1,
            )
        );
    }

    #[test]
    fn test_zero() {
        assert_eq!(Scalar::ZERO, Scalar::zero());
        assert_eq!(Scalar::ZERO, from_const(0));
        assert_eq!(Scalar::ZERO + from_const(1), from_const(1));
        assert_eq!(
            Scalar::ZERO * from_coefficients(1, 2, 3, 4, 5, 6, 7, 8),
            Scalar::ZERO
        );
    }

    #[test]
    fn test_one() {
        assert_eq!(Scalar::ONE, Scalar::one());
        assert_eq!(Scalar::ONE, from_const(1));
        assert_eq!(
            Scalar::ONE * from_coefficients(1, 2, 3, 4, 5, 6, 7, 8),
            from_coefficients(1, 2, 3, 4, 5, 6, 7, 8)
        );
    }

    #[test]
    fn test_max() {
        assert_eq!(Scalar::MAX.to_u256(), modulus_pow8() - U256::one());
    }

    #[test]
    fn test_multiplicative_generator() {
        assert_eq!(
            Scalar::MULTIPLICATIVE_GENERATOR,
            from_coefficients(
                0x3426065a, 0x13ac7c06, 0x023b1b71, 0x642192ae, 0x68087c79, 0x5481f83a, 0x5938d3c3,
                0x31ed2f75,
            )
        );

        let base_generator = Scalar::from(base::Scalar::MULTIPLICATIVE_GENERATOR);
        assert_eq!(
            base_generator.pow(from_const(MODULUS as u64 - 1)),
            Scalar::ONE,
            "the base field's generator has order p - 1, so it cannot generate the extension"
        );

        let kb2_generator = Scalar::from(kb2::Scalar::MULTIPLICATIVE_GENERATOR);
        assert_eq!(
            kb2_generator.pow(from_const((MODULUS as u64) * (MODULUS as u64) - 1)),
            Scalar::ONE,
            "the KB2 generator has order p^2 - 1, so it cannot generate the extension"
        );

        let kb4_generator = Scalar::from(kb4::Scalar::MULTIPLICATIVE_GENERATOR);
        let modulus = MODULUS as u128;
        assert_eq!(
            kb4_generator.pow(Scalar::from(modulus * modulus * modulus * modulus - 1)),
            Scalar::ONE,
            "the KB4 generator has order p^4 - 1, so it cannot generate the extension"
        );

        let order = modulus_pow8() - U256::one();
        let t = Scalar::try_from(order >> Scalar::S).unwrap();
        assert_eq!(
            Scalar::MULTIPLICATIVE_GENERATOR.pow(t),
            Scalar::ROOT_OF_UNITY
        );
        assert_ne!(
            Scalar::MULTIPLICATIVE_GENERATOR.pow(from_const(1u64 << Scalar::S)),
            Scalar::ONE
        );
    }

    #[test]
    fn test_minus_two() {
        assert_ne!(Scalar::MINUS_TWO, -from_const(2));

        let value = from_coefficients(7, 11, 13, 17, 19, 23, 29, 31);
        assert_eq!(value.invert_unwrap(), value.pow(Scalar::MINUS_TWO));
    }

    #[test]
    fn test_two_inv() {
        assert_eq!(Scalar::TWO_INV, from_const(2).invert_unwrap());
        assert_eq!(Scalar::TWO_INV.invert_unwrap(), from_const(2));
    }

    #[test]
    fn test_root_of_unity() {
        assert_eq!(
            Scalar::ROOT_OF_UNITY.square(),
            kb4::Scalar::ROOT_OF_UNITY.into()
        );
        assert_eq!(Scalar::S, kb4::Scalar::S + 1);
        for i in 0..Scalar::S {
            assert_ne!(
                Scalar::ROOT_OF_UNITY.pow(from_const(1u64 << i)),
                Scalar::ONE
            );
        }
        assert_eq!(
            Scalar::ROOT_OF_UNITY.pow(from_const(1u64 << Scalar::S)),
            Scalar::ONE
        );
    }

    #[test]
    fn test_root_of_unity_inverse() {
        assert_eq!(
            Scalar::ROOT_OF_UNITY_INV,
            Scalar::ROOT_OF_UNITY.invert_unwrap()
        );
    }

    #[test]
    fn test_delta() {
        assert_eq!(
            Scalar::DELTA,
            Scalar::MULTIPLICATIVE_GENERATOR.pow(from_const(1u64 << Scalar::S))
        );
    }

    #[test]
    fn test_equality() {
        assert_eq!(
            from_coefficients(1, 2, 3, 4, 5, 6, 7, 8),
            from_coefficients(1, 2, 3, 4, 5, 6, 7, 8)
        );
        assert_ne!(
            from_coefficients(1, 2, 3, 4, 5, 6, 7, 8),
            from_coefficients(1, 2, 3, 4, 5, 6, 7, 9)
        );
    }

    #[test]
    fn test_total_order() {
        let v0 = from_coefficients(0, 0, 0, 0, 0, 0, 0, 0);
        let v1 = from_coefficients(0, 0, 0, 0, 0, 0, 0, 1);
        let v2 = from_coefficients(0, 0, 0, 0, 0, 0, 1, 0);
        let v3 = from_coefficients(0, 0, 0, 0, 0, 1, 0, 0);
        let v4 = from_coefficients(0, 0, 0, 0, 1, 0, 0, 0);
        let v5 = from_coefficients(0, 0, 0, 1, 0, 0, 0, 0);
        let v6 = from_coefficients(0, 0, 1, 0, 0, 0, 0, 0);
        let v7 = from_coefficients(0, 1, 0, 0, 0, 0, 0, 0);
        let v8 = from_coefficients(1, 0, 0, 0, 0, 0, 0, 0);

        assert_eq!(v0.cmp(&v0), Ordering::Equal);
        assert_eq!(v0.cmp(&v1), Ordering::Less);
        assert_eq!(v1.cmp(&v2), Ordering::Less);
        assert_eq!(v2.cmp(&v3), Ordering::Less);
        assert_eq!(v3.cmp(&v4), Ordering::Less);
        assert_eq!(v4.cmp(&v5), Ordering::Less);
        assert_eq!(v5.cmp(&v6), Ordering::Less);
        assert_eq!(v6.cmp(&v7), Ordering::Less);
        assert_eq!(v7.cmp(&v8), Ordering::Less);
        assert_eq!(v8.cmp(&v7), Ordering::Greater);
        assert_eq!(v8.cmp(&v8), Ordering::Equal);
    }

    #[test]
    fn test_ct_eq() {
        let a = from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);
        let b = from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);
        let c = from_coefficients(1, 2, 3, 4, 5, 6, 7, 9);
        let d = from_coefficients(9, 2, 3, 4, 5, 6, 7, 8);
        assert_eq!(bool::from(a.ct_eq(&b)), true);
        assert_eq!(bool::from(a.ct_eq(&c)), false);
        assert_eq!(bool::from(a.ct_eq(&d)), false);
    }

    #[test]
    fn test_ct_gt() {
        let v0 = from_coefficients(0, 0, 0, 0, 0, 0, 0, 0);
        let v1 = from_coefficients(0, 0, 0, 0, 0, 0, 0, 42);
        let v2 = from_coefficients(0, 0, 0, 0, 0, 0, 1, 0);
        assert_eq!(bool::from(v0.ct_gt(&v0)), false);
        assert_eq!(bool::from(v1.ct_gt(&v0)), true);
        assert_eq!(bool::from(v2.ct_gt(&v1)), true);
        assert_eq!(bool::from(v0.ct_gt(&v2)), false);
    }

    #[test]
    fn test_ct_lt() {
        let v0 = from_coefficients(0, 0, 0, 0, 0, 0, 0, 0);
        let v1 = from_coefficients(0, 0, 0, 0, 0, 0, 0, 42);
        let v2 = from_coefficients(0, 0, 0, 0, 0, 0, 1, 0);
        assert_eq!(bool::from(v0.ct_lt(&v1)), true);
        assert_eq!(bool::from(v1.ct_lt(&v2)), true);
        assert_eq!(bool::from(v2.ct_lt(&v0)), false);
    }

    #[test]
    fn test_conditional_select() {
        let a = from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);
        let b = from_coefficients(9, 10, 11, 12, 13, 14, 15, 16);
        assert_eq!(Scalar::conditional_select(&a, &b, Choice::from(0)), a);
        assert_eq!(Scalar::conditional_select(&a, &b, Choice::from(1)), b);
    }

    #[test]
    fn test_add() {
        let lhs = from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);
        let rhs = from_coefficients(9, 10, 11, 12, 13, 14, 15, 16);
        let expected = from_coefficients(10, 12, 14, 16, 18, 20, 22, 24);
        assert_eq!(lhs + rhs, expected);
        assert_eq!(lhs + &rhs, expected);
    }

    #[test]
    fn test_add_wraparound() {
        let lhs = from_coefficients(
            MODULUS - 1,
            MODULUS - 2,
            MODULUS - 3,
            MODULUS - 4,
            MODULUS - 5,
            MODULUS - 6,
            MODULUS - 7,
            MODULUS - 8,
        );
        let rhs = from_coefficients(2, 3, 4, 5, 6, 7, 8, 9);
        assert_eq!(lhs + rhs, from_coefficients(1, 1, 1, 1, 1, 1, 1, 1));
    }

    #[test]
    fn test_add_assign() {
        let mut lhs = from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);
        lhs += from_coefficients(9, 10, 11, 12, 13, 14, 15, 16);
        assert_eq!(lhs, from_coefficients(10, 12, 14, 16, 18, 20, 22, 24));
    }

    #[test]
    fn test_add_assign_ref() {
        let mut lhs = from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);
        lhs += &from_coefficients(9, 10, 11, 12, 13, 14, 15, 16);
        assert_eq!(lhs, from_coefficients(10, 12, 14, 16, 18, 20, 22, 24));
    }

    #[test]
    fn test_add_base_scalar() {
        let lhs = from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);
        let rhs = base::Scalar::from_const(5);
        let expected = from_coefficients(1, 2, 3, 4, 5, 6, 7, 13);
        assert_eq!(lhs + rhs, expected);
        assert_eq!(lhs + &rhs, expected);
    }

    #[test]
    fn test_add_assign_base_scalar() {
        let rhs = base::Scalar::from_const(5);
        let expected = from_coefficients(1, 2, 3, 4, 5, 6, 7, 13);

        let mut lhs = from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);
        lhs += rhs;
        assert_eq!(lhs, expected);

        let mut lhs = from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);
        lhs += &rhs;
        assert_eq!(lhs, expected);
    }

    #[test]
    fn test_add_kb2() {
        let lhs = from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);
        let rhs = kb2::Scalar::from_coefficients(5, 6);
        let expected = from_coefficients(1, 2, 3, 4, 5, 6, 12, 14);
        assert_eq!(lhs + rhs, expected);
        assert_eq!(lhs + &rhs, expected);
    }

    #[test]
    fn test_add_assign_kb2() {
        let rhs = kb2::Scalar::from_coefficients(5, 6);
        let expected = from_coefficients(1, 2, 3, 4, 5, 6, 12, 14);

        let mut lhs = from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);
        lhs += rhs;
        assert_eq!(lhs, expected);

        let mut lhs = from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);
        lhs += &rhs;
        assert_eq!(lhs, expected);
    }

    #[test]
    fn test_add_kb4() {
        let lhs = from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);
        let rhs = kb4::Scalar::from_coefficients(5, 6, 7, 8);
        let expected = from_coefficients(1, 2, 3, 4, 10, 12, 14, 16);
        assert_eq!(lhs + rhs, expected);
        assert_eq!(lhs + &rhs, expected);
    }

    #[test]
    fn test_add_assign_kb4() {
        let rhs = kb4::Scalar::from_coefficients(5, 6, 7, 8);
        let expected = from_coefficients(1, 2, 3, 4, 10, 12, 14, 16);

        let mut lhs = from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);
        lhs += rhs;
        assert_eq!(lhs, expected);

        let mut lhs = from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);
        lhs += &rhs;
        assert_eq!(lhs, expected);
    }

    #[test]
    fn test_neg() {
        assert_eq!(-Scalar::ZERO, Scalar::ZERO);
        assert_eq!(
            -from_coefficients(1, 2, 3, 4, 5, 6, 7, 8),
            from_coefficients(
                MODULUS - 1,
                MODULUS - 2,
                MODULUS - 3,
                MODULUS - 4,
                MODULUS - 5,
                MODULUS - 6,
                MODULUS - 7,
                MODULUS - 8,
            )
        );
        assert_eq!(
            from_coefficients(1, 2, 3, 4, 5, 6, 7, 8) + -from_coefficients(1, 2, 3, 4, 5, 6, 7, 8),
            Scalar::ZERO
        );
    }

    #[test]
    fn test_sub() {
        let lhs = from_coefficients(10, 12, 14, 16, 18, 20, 22, 24);
        let rhs = from_coefficients(9, 10, 11, 12, 13, 14, 15, 16);
        let expected = from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);
        assert_eq!(lhs - rhs, expected);
        assert_eq!(lhs - &rhs, expected);
    }

    #[test]
    fn test_sub_wraparound() {
        let lhs = from_coefficients(1, 1, 1, 1, 1, 1, 1, 1);
        let rhs = from_coefficients(2, 3, 4, 5, 6, 7, 8, 9);
        assert_eq!(
            lhs - rhs,
            from_coefficients(
                MODULUS - 1,
                MODULUS - 2,
                MODULUS - 3,
                MODULUS - 4,
                MODULUS - 5,
                MODULUS - 6,
                MODULUS - 7,
                MODULUS - 8,
            )
        );
    }

    #[test]
    fn test_sub_assign() {
        let mut lhs = from_coefficients(10, 12, 14, 16, 18, 20, 22, 24);
        lhs -= from_coefficients(9, 10, 11, 12, 13, 14, 15, 16);
        assert_eq!(lhs, from_coefficients(1, 2, 3, 4, 5, 6, 7, 8));
    }

    #[test]
    fn test_sub_assign_ref() {
        let mut lhs = from_coefficients(10, 12, 14, 16, 18, 20, 22, 24);
        lhs -= &from_coefficients(9, 10, 11, 12, 13, 14, 15, 16);
        assert_eq!(lhs, from_coefficients(1, 2, 3, 4, 5, 6, 7, 8));
    }

    #[test]
    fn test_sub_base_scalar() {
        let lhs = from_coefficients(1, 2, 3, 4, 5, 6, 7, 13);
        let rhs = base::Scalar::from_const(5);
        let expected = from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);
        assert_eq!(lhs - rhs, expected);
        assert_eq!(lhs - &rhs, expected);
    }

    #[test]
    fn test_sub_assign_base_scalar() {
        let rhs = base::Scalar::from_const(5);
        let expected = from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);

        let mut lhs = from_coefficients(1, 2, 3, 4, 5, 6, 7, 13);
        lhs -= rhs;
        assert_eq!(lhs, expected);

        let mut lhs = from_coefficients(1, 2, 3, 4, 5, 6, 7, 13);
        lhs -= &rhs;
        assert_eq!(lhs, expected);
    }

    #[test]
    fn test_sub_kb2() {
        let lhs = from_coefficients(1, 2, 3, 4, 5, 6, 12, 14);
        let rhs = kb2::Scalar::from_coefficients(5, 6);
        let expected = from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);
        assert_eq!(lhs - rhs, expected);
        assert_eq!(lhs - &rhs, expected);
    }

    #[test]
    fn test_sub_assign_kb2() {
        let rhs = kb2::Scalar::from_coefficients(5, 6);
        let expected = from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);

        let mut lhs = from_coefficients(1, 2, 3, 4, 5, 6, 12, 14);
        lhs -= rhs;
        assert_eq!(lhs, expected);

        let mut lhs = from_coefficients(1, 2, 3, 4, 5, 6, 12, 14);
        lhs -= &rhs;
        assert_eq!(lhs, expected);
    }

    #[test]
    fn test_sub_kb4() {
        let lhs = from_coefficients(1, 2, 3, 4, 10, 12, 14, 16);
        let rhs = kb4::Scalar::from_coefficients(5, 6, 7, 8);
        let expected = from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);
        assert_eq!(lhs - rhs, expected);
        assert_eq!(lhs - &rhs, expected);
    }

    #[test]
    fn test_sub_assign_kb4() {
        let rhs = kb4::Scalar::from_coefficients(5, 6, 7, 8);
        let expected = from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);

        let mut lhs = from_coefficients(1, 2, 3, 4, 10, 12, 14, 16);
        lhs -= rhs;
        assert_eq!(lhs, expected);

        let mut lhs = from_coefficients(1, 2, 3, 4, 10, 12, 14, 16);
        lhs -= &rhs;
        assert_eq!(lhs, expected);
    }

    #[test]
    fn test_extension_root() {
        let z = from_coefficients(0, 0, 0, 1, 0, 0, 0, 0);
        let z_squared = z * z;
        // Z^2 = Y, i.e. the KB4 generator embedded with a zero Z-coefficient.
        assert_eq!(z_squared, from_coefficients(0, 0, 0, 0, 0, 1, 0, 0));
        assert_eq!(z * &z, z_squared);
        // Z^4 = X, i.e. the KB2 generator.
        let z_pow4 = z_squared * z_squared;
        assert_eq!(z_pow4, from_coefficients(0, 0, 0, 0, 0, 0, 1, 0));
        // Z^8 = QUADRATIC_NON_RESIDUE.
        assert_eq!(z_pow4 * z_pow4, from_const(QUADRATIC_NON_RESIDUE as u64));
    }

    #[test]
    fn test_mul_by_zero() {
        let value = from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);
        assert_eq!(Scalar::ZERO * value, Scalar::ZERO);
        assert_eq!(value * Scalar::ZERO, Scalar::ZERO);
    }

    #[test]
    fn test_mul_by_one() {
        let value = from_coefficients(2, 3, 4, 5, 6, 7, 8, 9);
        assert_eq!(Scalar::ONE * value, value);
        assert_eq!(value * Scalar::ONE, value);
    }

    #[test]
    fn test_mul() {
        let lhs = from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);
        let rhs = from_coefficients(9, 10, 11, 12, 13, 14, 15, 16);
        let expected = from_coefficients(408, 756, 542, 958, 499, 937, 689, 1187);
        assert_eq!(lhs * rhs, expected);
        assert_eq!(lhs * &rhs, expected);
        assert_eq!(rhs * lhs, expected);
    }

    #[test]
    fn test_mul_assign() {
        let mut lhs = from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);
        lhs *= from_coefficients(9, 10, 11, 12, 13, 14, 15, 16);
        assert_eq!(
            lhs,
            from_coefficients(408, 756, 542, 958, 499, 937, 689, 1187)
        );
    }

    #[test]
    fn test_mul_assign_ref() {
        let mut lhs = from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);
        lhs *= &from_coefficients(9, 10, 11, 12, 13, 14, 15, 16);
        assert_eq!(
            lhs,
            from_coefficients(408, 756, 542, 958, 499, 937, 689, 1187)
        );
    }

    #[test]
    fn test_mul_base_scalar() {
        let lhs = from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);
        let rhs = base::Scalar::from_const(5);
        let expected = from_coefficients(5, 10, 15, 20, 25, 30, 35, 40);
        assert_eq!(lhs * rhs, expected);
        assert_eq!(lhs * &rhs, expected);
    }

    #[test]
    fn test_mul_assign_base_scalar() {
        let rhs = base::Scalar::from_const(5);
        let expected = from_coefficients(5, 10, 15, 20, 25, 30, 35, 40);

        let mut lhs = from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);
        lhs *= rhs;
        assert_eq!(lhs, expected);

        let mut lhs = from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);
        lhs *= &rhs;
        assert_eq!(lhs, expected);
    }

    #[test]
    fn test_mul_kb2() {
        let lhs = from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);
        let rhs = kb2::Scalar::from_coefficients(5, 6);
        let expected = from_coefficients(16, 27, 38, 69, 60, 111, 82, 153);
        assert_eq!(lhs * rhs, expected);
        assert_eq!(lhs * &rhs, expected);
    }

    #[test]
    fn test_mul_assign_kb2() {
        let rhs = kb2::Scalar::from_coefficients(5, 6);
        let expected = from_coefficients(16, 27, 38, 69, 60, 111, 82, 153);

        let mut lhs = from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);
        lhs *= rhs;
        assert_eq!(lhs, expected);

        let mut lhs = from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);
        lhs *= &rhs;
        assert_eq!(lhs, expected);
    }

    #[test]
    fn test_mul_kb4() {
        let lhs = from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);
        let rhs = kb4::Scalar::from_coefficients(5, 6, 7, 8);
        let expected = from_coefficients(60, 106, 79, 143, 164, 306, 223, 391);
        assert_eq!(lhs * rhs, expected);
        assert_eq!(lhs * &rhs, expected);
    }

    #[test]
    fn test_mul_assign_kb4() {
        let rhs = kb4::Scalar::from_coefficients(5, 6, 7, 8);
        let expected = from_coefficients(60, 106, 79, 143, 164, 306, 223, 391);

        let mut lhs = from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);
        lhs *= rhs;
        assert_eq!(lhs, expected);

        let mut lhs = from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);
        lhs *= &rhs;
        assert_eq!(lhs, expected);
    }

    #[test]
    fn test_div_by_one() {
        let value = from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);
        assert_eq!(value / Scalar::ONE, value);
        assert_eq!(value / &Scalar::ONE, value);
    }

    #[test]
    fn test_div() {
        let lhs = from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);
        let rhs = from_coefficients(9, 10, 11, 12, 13, 14, 15, 16);
        assert_eq!((lhs * rhs) / rhs, lhs);
        assert_eq!((lhs * rhs) / &rhs, lhs);
    }

    #[test]
    fn test_div_assign() {
        let rhs = from_coefficients(9, 10, 11, 12, 13, 14, 15, 16);
        let mut lhs = from_coefficients(1, 2, 3, 4, 5, 6, 7, 8) * rhs;
        lhs /= rhs;
        assert_eq!(lhs, from_coefficients(1, 2, 3, 4, 5, 6, 7, 8));
    }

    #[test]
    fn test_div_assign_ref() {
        let rhs = from_coefficients(9, 10, 11, 12, 13, 14, 15, 16);
        let mut lhs = from_coefficients(1, 2, 3, 4, 5, 6, 7, 8) * rhs;
        lhs /= &rhs;
        assert_eq!(lhs, from_coefficients(1, 2, 3, 4, 5, 6, 7, 8));
    }

    #[test]
    fn test_div_base_scalar() {
        let lhs = from_coefficients(5, 10, 15, 20, 25, 30, 35, 40);
        let rhs = base::Scalar::from_const(5);
        let expected = from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);
        assert_eq!(lhs / rhs, expected);
        assert_eq!(lhs / &rhs, expected);
    }

    #[test]
    fn test_div_assign_base_scalar() {
        let rhs = base::Scalar::from_const(5);
        let expected = from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);

        let mut lhs = from_coefficients(5, 10, 15, 20, 25, 30, 35, 40);
        lhs /= rhs;
        assert_eq!(lhs, expected);

        let mut lhs = from_coefficients(5, 10, 15, 20, 25, 30, 35, 40);
        lhs /= &rhs;
        assert_eq!(lhs, expected);
    }

    #[test]
    fn test_div_kb2() {
        let lhs = from_coefficients(16, 27, 38, 69, 60, 111, 82, 153);
        let rhs = kb2::Scalar::from_coefficients(5, 6);
        let expected = from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);
        assert_eq!(lhs / rhs, expected);
        assert_eq!(lhs / &rhs, expected);
    }

    #[test]
    fn test_div_assign_kb2() {
        let rhs = kb2::Scalar::from_coefficients(5, 6);
        let expected = from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);

        let mut lhs = from_coefficients(16, 27, 38, 69, 60, 111, 82, 153);
        lhs /= rhs;
        assert_eq!(lhs, expected);

        let mut lhs = from_coefficients(16, 27, 38, 69, 60, 111, 82, 153);
        lhs /= &rhs;
        assert_eq!(lhs, expected);
    }

    #[test]
    fn test_div_kb4() {
        let lhs = from_coefficients(60, 106, 79, 143, 164, 306, 223, 391);
        let rhs = kb4::Scalar::from_coefficients(5, 6, 7, 8);
        let expected = from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);
        assert_eq!(lhs / rhs, expected);
        assert_eq!(lhs / &rhs, expected);
    }

    #[test]
    fn test_div_assign_kb4() {
        let rhs = kb4::Scalar::from_coefficients(5, 6, 7, 8);
        let expected = from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);

        let mut lhs = from_coefficients(60, 106, 79, 143, 164, 306, 223, 391);
        lhs /= rhs;
        assert_eq!(lhs, expected);

        let mut lhs = from_coefficients(60, 106, 79, 143, 164, 306, 223, 391);
        lhs /= &rhs;
        assert_eq!(lhs, expected);
    }

    fn test_inversion_impl(value: Scalar) {
        assert_ne!(value, Scalar::ZERO);
        assert_eq!(value * value.invert().unwrap(), Scalar::ONE);
        assert_eq!(value * value.invert_unwrap(), Scalar::ONE);
        assert_eq!(value * value.invert_or_zero(), Scalar::ONE);
        assert_eq!(value * value.invert_vartime().unwrap(), Scalar::ONE);
    }

    #[test]
    fn test_inversion() {
        assert!(Scalar::ZERO.invert_vartime().is_none());
        assert_eq!(Scalar::ZERO.invert_or_zero(), Scalar::ZERO);
        assert!(bool::from(Scalar::ZERO.invert().is_none()));
        test_inversion_impl(Scalar::ONE);
        test_inversion_impl(from_coefficients(0, 0, 0, 0, 0, 0, 0, 42));
        test_inversion_impl(from_coefficients(1, 0, 0, 0, 0, 0, 0, 0));
        test_inversion_impl(from_coefficients(0, 0, 0, 1, 0, 0, 0, 0));
        test_inversion_impl(from_coefficients(0, 0, 0, 0, 0, 1, 0, 0));
        test_inversion_impl(from_coefficients(0, 0, 0, 0, 0, 0, 1, 0));
        test_inversion_impl(from_coefficients(7, 11, 13, 17, 19, 23, 29, 31));
        test_inversion_impl(Scalar::MAX);
    }

    #[test]
    fn test_invert_batch() {
        let values = vec![
            from_coefficients(1, 2, 3, 4, 5, 6, 7, 8),
            from_coefficients(0, 0, 0, 0, 0, 0, 0, 42),
            Scalar::ONE,
            from_coefficients(3, 5, 7, 9, 11, 13, 15, 17),
        ];
        let expected: Vec<Scalar> = values
            .iter()
            .map(|value| value.invert_vartime().unwrap())
            .collect();

        let mut batch = values.clone();
        Scalar::invert_batch(&mut batch);
        assert_eq!(batch, expected);

        let mut batch = values;
        Scalar::invert_batch_vartime(&mut batch);
        assert_eq!(batch, expected);
    }

    #[test]
    fn test_sum() {
        let values = vec![
            from_coefficients(1, 2, 3, 4, 5, 6, 7, 8),
            from_coefficients(9, 10, 11, 12, 13, 14, 15, 16),
            from_coefficients(1, 1, 1, 1, 1, 1, 1, 1),
        ];
        let expected = from_coefficients(11, 13, 15, 17, 19, 21, 23, 25);
        assert_eq!(values.iter().sum::<Scalar>(), expected);
        assert_eq!(values.into_iter().sum::<Scalar>(), expected);
    }

    #[test]
    fn test_product() {
        let values = vec![
            from_coefficients(0, 0, 0, 0, 0, 0, 0, 2),
            from_coefficients(0, 0, 0, 0, 0, 0, 0, 3),
            from_coefficients(0, 0, 0, 0, 0, 0, 0, 4),
        ];
        let expected = from_coefficients(0, 0, 0, 0, 0, 0, 0, 24);
        assert_eq!(values.iter().product::<Scalar>(), expected);
        assert_eq!(values.into_iter().product::<Scalar>(), expected);
    }

    #[test]
    fn test_fmt_display() {
        assert_eq!(
            format!("{}", from_const(0)),
            "0x0000000000000000000000000000000000000000000000000000000000000000"
        );
        assert_eq!(
            format!("{}", from_const(0x13371337)),
            "0x0000000000000000000000000000000000000000000000000000000013371337"
        );
        assert_eq!(
            format!("{}", from_coefficients(1, 2, 3, 4, 5, 6, 7, 8)),
            "0x0000000001e4a5d47d1bd8a2b9f5a65ef27d5863623c821f027e0029ac000024"
        );
        assert_eq!(
            format!("{}", Scalar::MAX),
            "0x00f06e44682c2aa440f5f26a5ae174900568744cd653c806e41c0003f8000000"
        );
    }

    #[test]
    fn test_fmt_debug() {
        assert_eq!(
            format!("{:?}", from_const(0)),
            "Scalar(0x0000000000000000000000000000000000000000000000000000000000000000)"
        );
    }

    #[test]
    fn test_fmt_lower_hex() {
        assert_eq!(format!("{:x}", from_const(0x13371337)), "13371337");
        assert_eq!(format!("{:#x}", from_const(0x13371337)), "0x13371337");
        assert_eq!(
            format!("{:x}", from_coefficients(1, 2, 3, 4, 5, 6, 7, 8)),
            "1e4a5d47d1bd8a2b9f5a65ef27d5863623c821f027e0029ac000024"
        );
        assert_eq!(
            format!("{:x}", Scalar::MAX),
            "f06e44682c2aa440f5f26a5ae174900568744cd653c806e41c0003f8000000"
        );
    }

    #[test]
    fn test_fmt_upper_hex() {
        assert_eq!(format!("{:X}", from_const(0x13371337)), "13371337");
        assert_eq!(
            format!("{:X}", from_coefficients(1, 2, 3, 4, 5, 6, 7, 8)),
            "1E4A5D47D1BD8A2B9F5A65EF27D5863623C821F027E0029AC000024"
        );
        assert_eq!(
            format!("{:X}", Scalar::MAX),
            "F06E44682C2AA440F5F26A5AE174900568744CD653C806E41C0003F8000000"
        );
    }

    #[test]
    fn test_fmt_binary() {
        assert_eq!(format!("{:b}", from_const(0b1010)), "1010");
    }

    #[test]
    fn test_fmt_octal() {
        assert_eq!(format!("{:o}", from_const(0o755)), "755");
    }

    #[test]
    fn test_from_str() {
        assert_eq!("0".parse::<Scalar>().unwrap(), Scalar::ZERO);
        assert_eq!("42".parse::<Scalar>().unwrap(), from_const(42));
        assert_eq!("0x2a".parse::<Scalar>().unwrap(), from_const(42));
        assert_eq!("0b101010".parse::<Scalar>().unwrap(), from_const(42));
        assert_eq!("0o52".parse::<Scalar>().unwrap(), from_const(42));
        assert_eq!(
            "0xf06e44682c2aa440f5f26a5ae174900568744cd653c806e41c0003f8000000"
                .parse::<Scalar>()
                .unwrap(),
            Scalar::MAX
        );
    }

    #[test]
    fn test_from_str_invalid() {
        assert!("".parse::<Scalar>().is_err());
        assert!("not a number".parse::<Scalar>().is_err());
        assert!(
            "0xf06e44682c2aa440f5f26a5ae174900568744cd653c806e41c0003f8000001"
                .parse::<Scalar>()
                .is_err()
        );
        // MODULUS^8, i.e. one past the largest representable value.
        assert!(
            "424804331891979973455971894938199991873140910988886521584080257519598960641"
                .parse::<Scalar>()
                .is_err()
        );
    }

    #[test]
    fn test_parse_scalar() {
        assert_eq!(parse_scalar("0x2a"), from_const(42));
    }

    #[test]
    fn test_display_from_str_roundtrip() {
        let value = from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);
        assert_eq!(format!("{}", value).parse::<Scalar>().unwrap(), value);
        assert_eq!(
            format!("{}", Scalar::MAX).parse::<Scalar>().unwrap(),
            Scalar::MAX
        );
    }

    #[test]
    fn test_from_u8() {
        assert_eq!(Scalar::from(0u8), from_const(0));
        assert_eq!(Scalar::from(u8::MAX), from_const(u8::MAX as u64));
    }

    #[test]
    fn test_from_u16() {
        assert_eq!(Scalar::from(0u16), from_const(0));
        assert_eq!(Scalar::from(u16::MAX), from_const(u16::MAX as u64));
    }

    #[test]
    fn test_from_u32() {
        assert_eq!(Scalar::from(0u32), from_const(0));
        assert_eq!(Scalar::from(u32::MAX), from_const(u32::MAX as u64));
        assert_eq!(
            Scalar::from(u32::MAX),
            from_coefficients(0, 0, 0, 0, 0, 0, 2, 33554429)
        );
    }

    #[test]
    fn test_from_u64() {
        assert_eq!(Scalar::from(0u64), from_const(0));
        assert_eq!(Scalar::from(u64::MAX), from_const(u64::MAX));
        assert_eq!(
            Scalar::from(MODULUS as u64),
            from_coefficients(0, 0, 0, 0, 0, 0, 1, 0)
        );
        assert_eq!(
            Scalar::from(u64::MAX),
            from_coefficients(0, 0, 0, 0, 0, 4, 134746136, 402124771)
        );
    }

    #[test]
    fn test_from_u128() {
        assert_eq!(Scalar::from(0u128), from_const(0));
        assert_eq!(Scalar::from(42u128), from_const(42));
        let modulus = MODULUS as u128;
        assert_eq!(
            Scalar::from(modulus * modulus),
            from_coefficients(0, 0, 0, 0, 0, 1, 0, 0)
        );
        // MODULUS^4, i.e. the smallest value needing all five lower words.
        assert_eq!(
            Scalar::from(modulus * modulus * modulus * modulus),
            from_coefficients(0, 0, 0, 1, 0, 0, 0, 0)
        );
        assert_eq!(
            Scalar::from(u128::MAX),
            from_coefficients(0, 0, 0, 16, 1086490451, 1472761332, 1664577053, 1111325835)
        );
    }

    #[test]
    fn test_from_base_scalar() {
        assert_eq!(
            Scalar::from(base::Scalar::from_const(42)),
            from_coefficients(0, 0, 0, 0, 0, 0, 0, 42)
        );
        assert_eq!(Scalar::from(base::Scalar::ZERO), Scalar::ZERO);
    }

    #[test]
    fn test_from_kb2_scalar() {
        assert_eq!(
            Scalar::from(kb2::Scalar::from_coefficients(5, 6)),
            from_coefficients(0, 0, 0, 0, 0, 0, 5, 6)
        );
        assert_eq!(Scalar::from(kb2::Scalar::ZERO), Scalar::ZERO);
    }

    #[test]
    fn test_from_kb4_scalar() {
        assert_eq!(
            Scalar::from(kb4::Scalar::from_coefficients(5, 6, 7, 8)),
            from_coefficients(0, 0, 0, 0, 5, 6, 7, 8)
        );
        assert_eq!(Scalar::from(kb4::Scalar::ZERO), Scalar::ZERO);
    }

    #[test]
    fn test_try_from_usize() {
        assert_eq!(Scalar::try_from(0usize).unwrap(), from_const(0));
        assert_eq!(Scalar::try_from(42usize).unwrap(), from_const(42));
    }

    #[test]
    fn test_try_from_u256() {
        assert_eq!(Scalar::try_from(U256::from(0)).unwrap(), from_const(0));
        assert_eq!(Scalar::try_from(U256::from(42)).unwrap(), from_const(42));
        assert_eq!(
            Scalar::try_from(modulus_pow8() - U256::one()).unwrap(),
            Scalar::MAX
        );
        assert!(Scalar::try_from(modulus_pow8()).is_err());
        assert!(Scalar::try_from(U256::MAX).is_err());
    }

    #[test]
    fn test_is_even() {
        assert!(bool::from(
            from_coefficients(0, 0, 0, 0, 0, 0, 0, 0).is_even()
        ));
        assert!(bool::from(
            from_coefficients(1, 1, 0, 0, 0, 0, 0, 0).is_even()
        ));
        assert!(!bool::from(
            from_coefficients(0, 0, 0, 0, 0, 0, 0, 1).is_even()
        ));
        assert!(!bool::from(
            from_coefficients(1, 0, 0, 0, 0, 0, 0, 0).is_even()
        ));
    }

    #[test]
    fn test_is_odd() {
        assert!(!bool::from(
            from_coefficients(0, 0, 0, 0, 0, 0, 0, 0).is_odd()
        ));
        assert!(!bool::from(
            from_coefficients(1, 1, 0, 0, 0, 0, 0, 0).is_odd()
        ));
        assert!(bool::from(
            from_coefficients(0, 0, 0, 0, 0, 0, 0, 1).is_odd()
        ));
        assert!(bool::from(
            from_coefficients(1, 0, 0, 0, 0, 0, 0, 0).is_odd()
        ));
    }

    struct OsRng;

    impl rand_core::TryRng for OsRng {
        type Error = getrandom::Error;

        fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Self::Error> {
            getrandom::fill(dest)
        }

        fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
            let mut bytes = [0u8; 4];
            getrandom::fill(&mut bytes)?;
            Ok(u32::from_le_bytes(bytes))
        }

        fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
            let mut bytes = [0u8; 8];
            getrandom::fill(&mut bytes)?;
            Ok(u64::from_le_bytes(bytes))
        }
    }

    impl rand_core::TryCryptoRng for OsRng {}

    #[test]
    fn test_try_random() {
        let mut rng = OsRng;
        assert_ne!(
            Scalar::try_random(&mut rng).unwrap(),
            Scalar::try_random(&mut rng).unwrap()
        );
    }

    #[test]
    fn test_random() {
        let mut rng = rand_core::UnwrapErr(OsRng);
        assert_ne!(Scalar::random(&mut rng), Scalar::random(&mut rng));
    }

    #[test]
    fn test_random_default() {
        assert_ne!(Scalar::random_default(), Scalar::random_default());
    }

    #[test]
    fn test_double() {
        assert_eq!(
            from_coefficients(1, 2, 3, 4, 5, 6, 7, 8).double(),
            from_coefficients(2, 4, 6, 8, 10, 12, 14, 16)
        );
    }

    #[test]
    fn test_square() {
        assert_eq!(
            from_coefficients(0, 0, 0, 0, 0, 0, 0, 5).square(),
            from_coefficients(0, 0, 0, 0, 0, 0, 0, 25)
        );
    }

    #[test]
    fn test_power() {
        let value = from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);
        assert_eq!(value.pow(Scalar::ZERO), Scalar::ONE);
        assert_eq!(value.pow(Scalar::ONE), value);
        assert_eq!(value.pow(from_const(2)), value * value);
        assert_eq!(value.pow(from_const(3)), value * value * value);
    }

    #[test]
    fn test_power_vartime() {
        let value = from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);
        assert_eq!(value.pow_vartime(Scalar::ZERO), Scalar::ONE);
        assert_eq!(value.pow_vartime(from_const(3)), value * value * value);
        assert_eq!(value.pow_vartime(from_const(3)), value.pow(from_const(3)));
    }

    #[test]
    fn test_integer_division() {
        assert_eq!(
            from_const(13).div_int(&from_const(5)),
            (from_const(2), from_const(3))
        );
    }

    #[test]
    fn test_integer_division_multi_word() {
        // lhs = 1*MODULUS^7 + 2*MODULUS^6 + ... + 7*MODULUS + 8, divided by 1000000007.
        let lhs = from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);
        let rhs = from_const(1000000007);
        let quotient = from_coefficients(
            0, 2, 278497010, 233583434, 74704322, 1534429383, 2040231177, 1038845133,
        );
        let remainder = from_const(109259737);
        assert_eq!(lhs.div_int(&rhs), (quotient, remainder));
    }

    #[test]
    fn test_try_from_le_bytes() {
        let mut bytes = [0u8; 32];
        bytes[0..8].copy_from_slice(&42u64.to_le_bytes());
        assert_eq!(Scalar::try_from_le_bytes(&bytes).unwrap(), from_const(42));
        assert!(bool::from(
            Scalar::try_from_le_bytes(&[255u8; 32]).is_none()
        ));
    }

    #[test]
    fn test_try_from_be_bytes() {
        let mut bytes = [0u8; 32];
        bytes[24..32].copy_from_slice(&42u64.to_be_bytes());
        assert_eq!(Scalar::try_from_be_bytes(&bytes).unwrap(), from_const(42));
        assert!(bool::from(
            Scalar::try_from_be_bytes(&[255u8; 32]).is_none()
        ));
    }

    #[test]
    fn test_le_be_bytes_roundtrip() {
        let value = from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);
        let n = value.to_u256();
        assert_eq!(
            Scalar::try_from_le_bytes(&n.to_little_endian()).unwrap(),
            value
        );
        assert_eq!(
            Scalar::try_from_be_bytes(&n.to_big_endian()).unwrap(),
            value
        );
    }

    #[test]
    fn test_try_to_u8() {
        assert_eq!(from_const(0).try_to_u8().unwrap(), 0);
        assert_eq!(from_const(u8::MAX as u64).try_to_u8().unwrap(), u8::MAX);
        assert!(from_const(u8::MAX as u64 + 1).try_to_u8().is_none());
        assert!(
            from_coefficients(1, 0, 0, 0, 0, 0, 0, 0)
                .try_to_u8()
                .is_none()
        );
    }

    #[test]
    fn test_try_to_u16() {
        assert_eq!(from_const(0).try_to_u16().unwrap(), 0);
        assert_eq!(from_const(u16::MAX as u64).try_to_u16().unwrap(), u16::MAX);
        assert!(from_const(u16::MAX as u64 + 1).try_to_u16().is_none());
        assert!(
            from_coefficients(1, 0, 0, 0, 0, 0, 0, 0)
                .try_to_u16()
                .is_none()
        );
    }

    #[test]
    fn test_field256_to_le_bytes() {
        let value = from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);
        assert_eq!(value.to_le_bytes(), value.to_u256().to_little_endian());
    }

    #[test]
    fn test_field256_to_be_bytes() {
        let value = from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);
        assert_eq!(value.to_be_bytes(), value.to_u256().to_big_endian());
    }

    #[test]
    fn test_field256_le_be_bytes_roundtrip() {
        let value = from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);
        assert_eq!(
            Scalar::try_from_le_bytes(&value.to_le_bytes()).unwrap(),
            value
        );
        assert_eq!(
            Scalar::try_from_be_bytes(&value.to_be_bytes()).unwrap(),
            value
        );
    }

    #[test]
    fn test_from_u512_mod_n() {
        assert_eq!(Scalar::from_u512_mod_n(U512::from(0)), from_const(0));
        assert_eq!(Scalar::from_u512_mod_n(U512::from(42)), from_const(42));
        assert_eq!(
            Scalar::from_u512_mod_n(U512::from(MODULUS)),
            from_coefficients(0, 0, 0, 0, 0, 0, 1, 0)
        );
        let modulus_pow8 = U512::from(modulus_pow8());
        assert_eq!(Scalar::from_u512_mod_n(modulus_pow8), Scalar::ZERO);
        assert_eq!(
            Scalar::from_u512_mod_n(modulus_pow8 + U512::from(1)),
            Scalar::ONE
        );
        assert_eq!(
            Scalar::from_u512_mod_n(U512::MAX),
            Scalar::try_from(U256::try_from(U512::MAX % modulus_pow8).unwrap()).unwrap()
        );
    }

    #[test]
    fn test_from_h512() {
        let mut bytes = [0u8; 64];
        bytes[0..8].copy_from_slice(&42u64.to_le_bytes());
        assert_eq!(Scalar::from_h512(H512::from_slice(&bytes)), from_const(42));
    }

    #[test]
    fn test_field256_try_to_u32() {
        assert_eq!(from_const(0).try_to_u32().unwrap(), 0);
        assert_eq!(from_const(u32::MAX as u64).try_to_u32().unwrap(), u32::MAX);
        assert_eq!(
            from_coefficients(0, 0, 0, 0, 0, 0, 1, 0)
                .try_to_u32()
                .unwrap(),
            MODULUS
        );
        assert!(bool::from(
            from_const(u32::MAX as u64 + 1).try_to_u32().is_none()
        ));
        assert!(bool::from(
            from_coefficients(1, 0, 0, 0, 0, 0, 0, 0)
                .try_to_u32()
                .is_none()
        ));
    }

    #[test]
    fn test_field256_try_to_u64() {
        assert_eq!(from_const(0).try_to_u64().unwrap(), 0);
        assert_eq!(from_const(42).try_to_u64().unwrap(), 42);
        assert_eq!(from_const(u64::MAX).try_to_u64().unwrap(), u64::MAX);
        assert_eq!(
            from_coefficients(0, 0, 0, 0, 0, 0, MODULUS - 1, MODULUS - 1)
                .try_to_u64()
                .unwrap(),
            (MODULUS as u64) * (MODULUS as u64) - 1
        );
        assert!(bool::from(
            from_coefficients(0, 0, 0, 0, 0, 5, 0, 0)
                .try_to_u64()
                .is_none()
        ));
        assert!(bool::from(
            from_coefficients(1, 0, 0, 0, 0, 0, 0, 0)
                .try_to_u64()
                .is_none()
        ));
    }

    #[test]
    fn test_field256_try_to_u128() {
        assert_eq!(from_const(0).try_to_u128().unwrap(), 0);
        assert_eq!(from_const(42).try_to_u128().unwrap(), 42);
        assert_eq!(Scalar::from(u128::MAX).try_to_u128().unwrap(), u128::MAX);
        assert_eq!(
            from_coefficients(
                0,
                0,
                0,
                0,
                MODULUS - 1,
                MODULUS - 1,
                MODULUS - 1,
                MODULUS - 1
            )
            .try_to_u128()
            .unwrap(),
            {
                let modulus = MODULUS as u128;
                modulus * modulus * modulus * modulus - 1
            }
        );
        assert!(bool::from(
            from_coefficients(0, 0, 0, 17, 0, 0, 0, 0)
                .try_to_u128()
                .is_none()
        ));
        assert!(bool::from(
            from_coefficients(0, 0, 1, 0, 0, 0, 0, 0)
                .try_to_u128()
                .is_none()
        ));
    }

    #[test]
    fn test_field256_to_u256() {
        assert_eq!(from_const(0).to_u256(), U256::from(0));
        assert_eq!(from_const(42).to_u256(), U256::from(42));
        assert_eq!(
            from_coefficients(1, 2, 3, 4, 5, 6, 7, 8).to_u256(),
            U256::from_str_radix(
                "1e4a5d47d1bd8a2b9f5a65ef27d5863623c821f027e0029ac000024",
                16
            )
            .unwrap()
        );
        assert_eq!(Scalar::MAX.to_u256(), modulus_pow8() - U256::one());
    }

    #[test]
    fn test_field256_to_u512() {
        assert_eq!(from_const(0).to_u512(), U512::from(0));
        assert_eq!(from_const(42).to_u512(), U512::from(42));
        assert_eq!(Scalar::MAX.to_u512(), U512::from(Scalar::MAX.to_u256()));
    }
}
