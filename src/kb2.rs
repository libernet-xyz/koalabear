use crate::base;
use crate::helpers::{
    CHARACTERS_LOWER_CASE, CHARACTERS_UPPER_CASE, MODULUS, QUADRATIC_NON_RESIDUE, kb_add, kb_add2,
    kb_from_montgomery, kb_mul, kb_mul2, kb_sub, kb_sub2, kb_to_montgomery,
};
use crate::kb4;
use crate::kb8;
use anyhow::anyhow;
use primitive_types::{U256, U512};
use rand_core::{CryptoRng, TryCryptoRng};
use starkom_ff::{Field, Field64};
use std::cmp::Ordering;
use std::fmt::{Binary, Debug, Display, Formatter, LowerHex, Octal, UpperHex};
use std::iter::{Product, Sum};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use std::str::FromStr;
use subtle::{
    Choice, ConditionallySelectable, ConstantTimeEq, ConstantTimeGreater, ConstantTimeLess,
    CtOption,
};

/// KoalaBear^2 extension field.
///
/// This is the degree-2 extension `GF(p)[X] / (X^2 - QUADRATIC_NON_RESIDUE)` of the KoalaBear
/// field, where `p` is [`MODULUS`]. A scalar `Scalar(a, b)` represents the polynomial `a * X + b`.
///
/// For all purposes other than field arithmetic (ordering, formatting, parsing, exponentiation,
/// etc.) a scalar is instead treated as the numeric value `a * MODULUS + b`. This gives every
/// scalar a canonical representative in `0..(MODULUS * MODULUS)` for those purposes.
///
/// NOTE: The `u32` words are stored in big-endian order: `Scalar::0` is the most significant and
/// `Scalar::1` is the least significant. Both coefficients are in Montgomery form, exactly like the
/// inner value of a [`base::Scalar`].
#[derive(Default, Copy, Clone, PartialEq, Eq)]
pub struct Scalar(pub(crate) u32, pub(crate) u32);

impl Scalar {
    /// Constructs a KoalaBear^2 scalar from the raw values of its coefficients, ie. `hi * X + lo`.
    ///
    /// Panics if either coefficient exceeds [`MODULUS`].
    #[inline]
    pub const fn from_coefficients(hi: u32, lo: u32) -> Self {
        Self(
            base::Scalar::from_const(hi).0,
            base::Scalar::from_const(lo).0,
        )
    }

    #[inline]
    const fn from_raw(value: u64) -> Self {
        const MODULUS_64: u64 = MODULUS as u64;
        assert!(value < MODULUS_64 * MODULUS_64, "invalid KoalaBear^2 value");
        Self::from_coefficients((value / MODULUS_64) as u32, (value % MODULUS_64) as u32)
    }

    /// Returns the raw (non-Montgomery) values of the coefficients.
    #[inline]
    const fn to_raw(&self) -> (u32, u32) {
        (kb_from_montgomery(self.0), kb_from_montgomery(self.1))
    }

    /// Constructs a KoalaBear^2 scalar from its raw 64-bit numeric value.
    ///
    /// Panics if the specified `value` exceeds `MODULUS^2`.
    #[inline]
    pub const fn from_const(value: u64) -> Self {
        Self::from_raw(value)
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
        ((self.0 == other.0) as u8 & (self.1 == other.1) as u8).into()
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
        let [hi, lo] = kb_add2([self.0, self.1], [rhs.0, rhs.1]);
        Self(hi, lo)
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
        [self.0, self.1] = kb_add2([self.0, self.1], [rhs.0, rhs.1]);
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
        Self(self.0, kb_add(self.1, rhs.0))
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
        self.1 = kb_add(self.1, rhs.0);
    }
}

impl<'a> AddAssign<&'a base::Scalar> for Scalar {
    fn add_assign(&mut self, rhs: &'a base::Scalar) {
        self.add_assign(*rhs);
    }
}

impl Add<kb4::Scalar> for Scalar {
    type Output = kb4::Scalar;

    fn add(self, rhs: kb4::Scalar) -> Self::Output {
        kb4::Scalar(rhs.0, rhs.1, kb_add(self.0, rhs.2), kb_add(self.1, rhs.3))
    }
}

impl<'a> Add<&'a kb4::Scalar> for Scalar {
    type Output = kb4::Scalar;

    fn add(self, rhs: &'a kb4::Scalar) -> Self::Output {
        self.add(*rhs)
    }
}

impl Add<kb8::Scalar> for Scalar {
    type Output = kb8::Scalar;

    fn add(self, rhs: kb8::Scalar) -> Self::Output {
        kb8::Scalar(
            rhs.0,
            rhs.1,
            rhs.2,
            rhs.3,
            rhs.4,
            rhs.5,
            kb_add(self.0, rhs.6),
            kb_add(self.1, rhs.7),
        )
    }
}

impl<'a> Add<&'a kb8::Scalar> for Scalar {
    type Output = kb8::Scalar;

    fn add(self, rhs: &'a kb8::Scalar) -> Self::Output {
        self.add(*rhs)
    }
}

impl Neg for Scalar {
    type Output = Scalar;

    fn neg(self) -> Self::Output {
        let [c0, c1] = kb_sub2([0, 0], [self.0, self.1]);
        Self(c0, c1)
    }
}

impl Sub<Self> for Scalar {
    type Output = Scalar;

    fn sub(self, rhs: Self) -> Self::Output {
        let [hi, lo] = kb_sub2([self.0, self.1], [rhs.0, rhs.1]);
        Self(hi, lo)
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
        [self.0, self.1] = kb_sub2([self.0, self.1], [rhs.0, rhs.1]);
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
        Self(self.0, kb_sub(self.1, rhs.0))
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
        self.1 = kb_sub(self.1, rhs.0);
    }
}

impl<'a> SubAssign<&'a base::Scalar> for Scalar {
    fn sub_assign(&mut self, rhs: &'a base::Scalar) {
        self.sub_assign(*rhs);
    }
}

impl Sub<kb4::Scalar> for Scalar {
    type Output = kb4::Scalar;

    fn sub(self, rhs: kb4::Scalar) -> Self::Output {
        kb4::Scalar(
            kb_sub(0, rhs.0),
            kb_sub(0, rhs.1),
            kb_sub(self.0, rhs.2),
            kb_sub(self.1, rhs.3),
        )
    }
}

impl<'a> Sub<&'a kb4::Scalar> for Scalar {
    type Output = kb4::Scalar;

    fn sub(self, rhs: &'a kb4::Scalar) -> Self::Output {
        self.sub(*rhs)
    }
}

impl Sub<kb8::Scalar> for Scalar {
    type Output = kb8::Scalar;

    fn sub(self, rhs: kb8::Scalar) -> Self::Output {
        kb8::Scalar(
            kb_sub(0, rhs.0),
            kb_sub(0, rhs.1),
            kb_sub(0, rhs.2),
            kb_sub(0, rhs.3),
            kb_sub(0, rhs.4),
            kb_sub(0, rhs.5),
            kb_sub(self.0, rhs.6),
            kb_sub(self.1, rhs.7),
        )
    }
}

impl<'a> Sub<&'a kb8::Scalar> for Scalar {
    type Output = kb8::Scalar;

    fn sub(self, rhs: &'a kb8::Scalar) -> Self::Output {
        self.sub(*rhs)
    }
}

impl Mul<Self> for Scalar {
    type Output = Scalar;

    fn mul(self, rhs: Self) -> Self::Output {
        let [hi, lo] = kb_mul2([self.0, self.1], [rhs.0, rhs.1]);
        Self(hi, lo)
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
        let [hi, lo] = kb_mul2([self.0, self.1], [rhs.0, rhs.1]);
        self.0 = hi;
        self.1 = lo;
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
        Self(kb_mul(self.0, rhs.0), kb_mul(self.1, rhs.0))
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
        self.0 = kb_mul(self.0, rhs.0);
        self.1 = kb_mul(self.1, rhs.0);
    }
}

impl<'a> MulAssign<&'a base::Scalar> for Scalar {
    fn mul_assign(&mut self, rhs: &'a base::Scalar) {
        self.mul_assign(*rhs);
    }
}

impl Mul<kb4::Scalar> for Scalar {
    type Output = kb4::Scalar;

    fn mul(self, rhs: kb4::Scalar) -> Self::Output {
        let [y0, y1] = kb_mul2([self.0, self.1], [rhs.0, rhs.1]);
        let [c0, c1] = kb_mul2([self.0, self.1], [rhs.2, rhs.3]);
        kb4::Scalar(y0, y1, c0, c1)
    }
}

impl<'a> Mul<&'a kb4::Scalar> for Scalar {
    type Output = kb4::Scalar;

    fn mul(self, rhs: &'a kb4::Scalar) -> Self::Output {
        self.mul(*rhs)
    }
}

impl Mul<kb8::Scalar> for Scalar {
    type Output = kb8::Scalar;

    fn mul(self, rhs: kb8::Scalar) -> Self::Output {
        let [a, b] = kb_mul2([self.0, self.1], [rhs.0, rhs.1]);
        let [c, d] = kb_mul2([self.0, self.1], [rhs.2, rhs.3]);
        let [e, f] = kb_mul2([self.0, self.1], [rhs.4, rhs.5]);
        let [g, h] = kb_mul2([self.0, self.1], [rhs.6, rhs.7]);
        kb8::Scalar(a, b, c, d, e, f, g, h)
    }
}

impl<'a> Mul<&'a kb8::Scalar> for Scalar {
    type Output = kb8::Scalar;

    fn mul(self, rhs: &'a kb8::Scalar) -> Self::Output {
        self.mul(*rhs)
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
        let inverse = rhs.invert_unwrap();
        Self(kb_mul(self.0, inverse.0), kb_mul(self.1, inverse.0))
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
        let inverse = rhs.invert_unwrap();
        self.0 = kb_mul(self.0, inverse.0);
        self.1 = kb_mul(self.1, inverse.0);
    }
}

impl<'a> DivAssign<&'a base::Scalar> for Scalar {
    fn div_assign(&mut self, rhs: &'a base::Scalar) {
        self.div_assign(*rhs);
    }
}

impl Div<kb4::Scalar> for Scalar {
    type Output = kb4::Scalar;

    fn div(self, rhs: kb4::Scalar) -> Self::Output {
        let inverse = rhs.invert_unwrap();
        let [y0, y1] = kb_mul2([self.0, self.1], [inverse.0, inverse.1]);
        let [c0, c1] = kb_mul2([self.0, self.1], [inverse.2, inverse.3]);
        kb4::Scalar(y0, y1, c0, c1)
    }
}

impl<'a> Div<&'a kb4::Scalar> for Scalar {
    type Output = kb4::Scalar;

    fn div(self, rhs: &'a kb4::Scalar) -> Self::Output {
        self.div(*rhs)
    }
}

impl Div<kb8::Scalar> for Scalar {
    type Output = kb8::Scalar;

    fn div(self, rhs: kb8::Scalar) -> Self::Output {
        self * rhs.invert_unwrap()
    }
}

impl<'a> Div<&'a kb8::Scalar> for Scalar {
    type Output = kb8::Scalar;

    fn div(self, rhs: &'a kb8::Scalar) -> Self::Output {
        self.div(*rhs)
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
        write!(f, "Scalar({:#018x})", self)
    }
}

impl Display for Scalar {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#018x}", self)
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
        Self::from_coefficients(0, value as u32)
    }
}

impl From<u16> for Scalar {
    fn from(value: u16) -> Self {
        Self::from_coefficients(0, value as u32)
    }
}

impl From<u32> for Scalar {
    fn from(value: u32) -> Self {
        Self::from_raw(value as u64)
    }
}

impl From<base::Scalar> for Scalar {
    fn from(value: base::Scalar) -> Self {
        Self(0, value.0)
    }
}

impl TryFrom<usize> for Scalar {
    type Error = anyhow::Error;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        Self::try_from(value as u64)
    }
}

impl TryFrom<u64> for Scalar {
    type Error = anyhow::Error;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        Self::try_from_le_bytes(&value.to_le_bytes())
            .into_option()
            .ok_or_else(|| anyhow!("{:#x} exceeds the KoalaBear^2 range", value))
    }
}

impl TryFrom<u128> for Scalar {
    type Error = anyhow::Error;

    fn try_from(value: u128) -> Result<Self, Self::Error> {
        if value > u64::MAX as u128 {
            Err(anyhow!("{:#x} exceeds the KoalaBear^2 range", value))
        } else {
            Self::try_from(value as u64)
        }
    }
}

impl TryFrom<U256> for Scalar {
    type Error = anyhow::Error;

    fn try_from(value: U256) -> Result<Self, Self::Error> {
        if value > U256::from(u64::MAX) {
            Err(anyhow!("{:#x} exceeds the KoalaBear^2 range", value))
        } else {
            Self::try_from(value.as_u64())
        }
    }
}

impl Field for Scalar {
    const MODULUS: &'static str = "0x3f010000fe000001";

    const CHARACTERISTIC: &'static str = "0x7f000001";

    const LEN: usize = 8;

    const ZERO: Self = Self(0, 0);

    const ONE: Self = Self::from_coefficients(0, 1);

    const MAX: Self = Self::from_coefficients(MODULUS - 1, MODULUS - 1);

    const S: usize = 25;

    const MULTIPLICATIVE_GENERATOR: Self = Self::from_coefficients(0x67480953, 0x7c9dd81f);

    const MINUS_TWO: Self = Self::from_coefficients(MODULUS - 1, MODULUS - 2);

    const TWO_INV: Self = Self::from_coefficients(0, 0x3f800001);

    const ROOT_OF_UNITY: Self = Self::from_coefficients(0x68dc4a89, 0);

    const ROOT_OF_UNITY_INV: Self = Self::from_coefficients(0x16e30df4, 0);

    const DELTA: Self = Self::from_coefficients(0x455fc4d3, 0x31c36f47);

    fn is_odd(&self) -> Choice {
        let (hi, lo) = self.to_raw();
        (((hi ^ lo) & 1) as u8).into()
    }

    fn try_random<R: TryCryptoRng>(rng: &mut R) -> Result<Self, R::Error> {
        Ok(Self(
            base::Scalar::try_random(rng)?.0,
            base::Scalar::try_random(rng)?.0,
        ))
    }

    fn random<R: CryptoRng>(rng: &mut R) -> Self {
        Self(base::Scalar::random(rng).0, base::Scalar::random(rng).0)
    }

    fn random_default() -> Self {
        Self(
            base::Scalar::random_default().0,
            base::Scalar::random_default().0,
        )
    }

    fn invert(&self) -> CtOption<Self> {
        let a = base::Scalar(self.0);
        let b = base::Scalar(self.1);
        let norm = b * b - a * a * base::Scalar::from_const(QUADRATIC_NON_RESIDUE);
        let conjugate = Self(kb_sub(0, self.0), self.1);
        norm.invert().map(|inverse_norm| conjugate * inverse_norm)
    }

    fn invert_vartime(&self) -> Option<Self> {
        let a = base::Scalar(self.0);
        let b = base::Scalar(self.1);
        let norm = b * b - a * a * base::Scalar::from_const(QUADRATIC_NON_RESIDUE);
        let conjugate = Self(kb_sub(0, self.0), self.1);
        norm.invert_vartime()
            .map(|inverse_norm| conjugate * inverse_norm)
    }

    fn pow(mut self, exp: Self) -> Self {
        let mut exponent = exp.to_u64();
        let mut result = Self::ONE;
        for _ in 0..Self::NUM_BITS {
            let product = result * self;
            result = Scalar::conditional_select(&result, &product, ((exponent & 1) as u8).into());
            exponent >>= 1;
            self = self.square();
        }
        result
    }

    fn pow_vartime(mut self, exp: Self) -> Self {
        let mut exponent = exp.to_u64();
        let mut result = Self::ONE;
        while exponent != 0 {
            if (exponent & 1) != 0 {
                result *= self;
            }
            exponent >>= 1;
            self = self.square();
        }
        result
    }

    fn div_int(&self, rhs: &Self) -> (Self, Self) {
        let lhs_value = self.to_u64();
        let rhs_value = rhs.to_u64();
        (
            Self::from_raw(lhs_value / rhs_value),
            Self::from_raw(lhs_value % rhs_value),
        )
    }

    fn try_from_le_bytes(bytes: &[u8]) -> CtOption<Self> {
        let mut fixed_bytes = [0u8; 8];
        fixed_bytes.copy_from_slice(bytes);
        let value = u64::from_le_bytes(fixed_bytes);
        let modulus = MODULUS as u64;
        let hi = value / modulus;
        let lo = value % modulus;
        CtOption::new(
            Self(kb_to_montgomery(hi as u32), kb_to_montgomery(lo as u32)),
            ((hi < modulus) as u8).into(),
        )
    }

    fn try_from_be_bytes(bytes: &[u8]) -> CtOption<Self> {
        let mut fixed_bytes = [0u8; 8];
        fixed_bytes.copy_from_slice(bytes);
        let value = u64::from_be_bytes(fixed_bytes);
        let modulus = MODULUS as u64;
        let hi = value / modulus;
        let lo = value % modulus;
        CtOption::new(
            Self(kb_to_montgomery(hi as u32), kb_to_montgomery(lo as u32)),
            ((hi < modulus) as u8).into(),
        )
    }

    fn from_str_radix(s: &str, radix: usize) -> Result<Self, std::fmt::Error> {
        assert!(radix >= 2 && radix <= 36);
        if s.is_empty() {
            return Err(std::fmt::Error);
        }
        let mut value: u64 = 0;
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
                .checked_mul(radix as u64)
                .ok_or(std::fmt::Error)?
                .checked_add(digit as u64)
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
        let mut value = self.to_u64();
        let mut s = String::default();
        let radix = radix as u64;
        while value != 0 {
            let digit = value % radix;
            s.push(characters[digit as usize] as char);
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
        let (hi, lo) = self.to_raw();
        if hi != 0 || lo > u8::MAX as u32 {
            None
        } else {
            Some(lo as u8)
        }
    }

    fn try_to_u16(&self) -> Option<u16> {
        let (hi, lo) = self.to_raw();
        if hi != 0 || lo > u16::MAX as u32 {
            None
        } else {
            Some(lo as u16)
        }
    }

    fn to_u256(&self) -> U256 {
        U256::from(self.to_u64())
    }

    fn to_u512(&self) -> U512 {
        U512::from(self.to_u64())
    }
}

impl Field64 for Scalar {
    fn to_le_bytes(&self) -> [u8; 8] {
        self.to_u64().to_le_bytes()
    }

    fn to_be_bytes(&self) -> [u8; 8] {
        self.to_u64().to_be_bytes()
    }

    fn from_u128_mod_n(u128: u128) -> Self {
        const MODULUS_U128: u128 = MODULUS as u128;
        Self::from_raw((u128 % (MODULUS_U128 * MODULUS_U128)) as u64)
    }

    fn from_u256_mod_n(u256: U256) -> Self {
        let modulus = U256::from(MODULUS);
        Self::from_raw((u256 % (modulus * modulus)).as_u64())
    }

    fn try_to_u32(&self) -> CtOption<u32> {
        let value = self.to_u64();
        CtOption::new(value as u32, ((value <= u32::MAX as u64) as u8).into())
    }

    fn to_u64(&self) -> u64 {
        let (hi, lo) = self.to_raw();
        (hi as u64) * (MODULUS as u64) + (lo as u64)
    }

    fn to_u128(&self) -> u128 {
        self.to_u64() as u128
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
    const fn from_coefficients(hi: u32, lo: u32) -> Scalar {
        Scalar::from_coefficients(hi, lo)
    }

    #[inline]
    fn parse_scalar(s: &'static str) -> Scalar {
        s.parse().unwrap()
    }

    #[test]
    fn test_from_const() {
        assert_eq!(from_const(0), Scalar::ZERO);
        assert_eq!(from_const(1), Scalar::ONE);
        assert_eq!(from_const(MODULUS as u64), from_coefficients(1, 0));
        assert_eq!(from_const(MODULUS as u64 + 1), from_coefficients(1, 1));
    }

    #[test]
    #[should_panic(expected = "invalid KoalaBear^2 value")]
    fn test_from_const_out_of_range() {
        from_const((MODULUS as u64) * (MODULUS as u64));
    }

    #[test]
    fn test_from_coefficients() {
        assert_eq!(from_coefficients(0, 0), Scalar::ZERO);
        assert_eq!(from_coefficients(0, 1), Scalar::ONE);
        assert_eq!(from_coefficients(1, 2), from_const(MODULUS as u64 + 2));
    }

    #[test]
    #[should_panic(expected = "invalid KoalaBear value")]
    fn test_from_coefficients_out_of_range() {
        from_coefficients(MODULUS, 0);
    }

    #[test]
    fn test_modulus() {
        assert_eq!(Scalar::MODULUS, "0x3f010000fe000001");
        assert_eq!(Scalar::MAX, from_coefficients(MODULUS - 1, MODULUS - 1));
    }

    #[test]
    fn test_zero() {
        assert_eq!(Scalar::ZERO, Scalar::zero());
        assert_eq!(Scalar::ZERO, from_const(0));
        assert_eq!(Scalar::ZERO + from_const(1), from_const(1));
        assert_eq!(Scalar::ZERO * from_const(42), Scalar::ZERO);
    }

    #[test]
    fn test_one() {
        assert_eq!(Scalar::ONE, Scalar::one());
        assert_eq!(Scalar::ONE, from_const(1));
        assert_eq!(Scalar::ONE * from_const(42), from_const(42));
    }

    #[test]
    fn test_max() {
        assert_eq!(Scalar::MAX, from_coefficients(MODULUS - 1, MODULUS - 1));
    }

    #[test]
    fn test_multiplicative_generator() {
        assert_eq!(
            Scalar::MULTIPLICATIVE_GENERATOR,
            from_coefficients(0x67480953, 0x7c9dd81f)
        );

        let base_generator = Scalar::from(base::Scalar::MULTIPLICATIVE_GENERATOR);
        assert_eq!(
            base_generator.pow(from_const(MODULUS as u64 - 1)),
            Scalar::ONE,
            "the base field's generator has order p - 1, so it cannot generate the extension"
        );

        let order = (MODULUS as u64) * (MODULUS as u64) - 1;
        let t = from_const(order >> Scalar::S);
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

        let value = from_coefficients(7, 11);
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
            base::Scalar::ROOT_OF_UNITY.into()
        );
        assert_eq!(Scalar::S, base::Scalar::S + 1);
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
        assert!(from_const(0) == from_const(0));
        assert!(from_const(0) != from_const(1));
        assert!(Scalar::MAX != Scalar::MAX - Scalar::ONE);
        assert!(Scalar::MAX == Scalar::MAX);
    }

    #[test]
    fn test_total_order() {
        let v0 = Scalar::ZERO;
        let v1 = Scalar::ONE;
        let v2 = from_coefficients(0, 42);
        let v3 = from_coefficients(1, 0);
        let v4 = Scalar::MAX;

        assert_eq!(v0.cmp(&v0), Ordering::Equal);
        assert_eq!(v0.cmp(&v1), Ordering::Less);
        assert_eq!(v1.cmp(&v2), Ordering::Less);
        assert_eq!(v2.cmp(&v3), Ordering::Less);
        assert_eq!(v3.cmp(&v4), Ordering::Less);
        assert_eq!(v4.cmp(&v3), Ordering::Greater);
        assert_eq!(v4.cmp(&v4), Ordering::Equal);
    }

    #[test]
    fn test_ct_eq() {
        let a = from_coefficients(1, 42);
        let b = from_coefficients(1, 42);
        let c = from_coefficients(1, 43);
        let d = from_coefficients(2, 42);
        assert_eq!(bool::from(a.ct_eq(&b)), true);
        assert_eq!(bool::from(a.ct_eq(&c)), false);
        assert_eq!(bool::from(a.ct_eq(&d)), false);
    }

    #[test]
    fn test_ct_gt() {
        let v0 = Scalar::ZERO;
        let v1 = from_coefficients(0, 42);
        let v2 = from_coefficients(1, 0);
        assert_eq!(bool::from(v0.ct_gt(&v0)), false);
        assert_eq!(bool::from(v1.ct_gt(&v0)), true);
        assert_eq!(bool::from(v2.ct_gt(&v1)), true);
        assert_eq!(bool::from(v0.ct_gt(&v2)), false);
    }

    #[test]
    fn test_ct_lt() {
        let v0 = Scalar::ZERO;
        let v1 = from_coefficients(0, 42);
        let v2 = from_coefficients(1, 0);
        assert_eq!(bool::from(v0.ct_lt(&v1)), true);
        assert_eq!(bool::from(v1.ct_lt(&v2)), true);
        assert_eq!(bool::from(v2.ct_lt(&v0)), false);
    }

    #[test]
    fn test_conditional_select() {
        let a = from_const(12);
        let b = from_const(34);
        assert_eq!(Scalar::conditional_select(&a, &b, Choice::from(0)), a);
        assert_eq!(Scalar::conditional_select(&a, &b, Choice::from(1)), b);
    }

    #[test]
    fn test_add() {
        let lhs = from_coefficients(10, 20);
        let rhs = from_coefficients(5, 7);
        assert_eq!(lhs + rhs, from_coefficients(15, 27));
        assert_eq!(lhs + &rhs, from_coefficients(15, 27));
    }

    #[test]
    fn test_add_wraparound() {
        let lhs = from_coefficients(MODULUS - 5, MODULUS - 3);
        let rhs = from_coefficients(10, 10);
        assert_eq!(lhs + rhs, from_coefficients(5, 7));
    }

    #[test]
    fn test_add_assign() {
        let mut lhs = from_coefficients(10, 20);
        lhs += from_coefficients(5, 7);
        assert_eq!(lhs, from_coefficients(15, 27));
    }

    #[test]
    fn test_add_assign_ref() {
        let mut lhs = from_coefficients(10, 20);
        lhs += &from_coefficients(5, 7);
        assert_eq!(lhs, from_coefficients(15, 27));
    }

    #[test]
    fn test_add_base_scalar() {
        let lhs = from_coefficients(1, 2);
        let rhs = base::Scalar::from_const(5);
        assert_eq!(lhs + rhs, from_coefficients(1, 7));
        assert_eq!(lhs + &rhs, from_coefficients(1, 7));
    }

    #[test]
    fn test_add_assign_base_scalar() {
        let rhs = base::Scalar::from_const(5);

        let mut lhs = from_coefficients(1, 2);
        lhs += rhs;
        assert_eq!(lhs, from_coefficients(1, 7));

        let mut lhs = from_coefficients(1, 2);
        lhs += &rhs;
        assert_eq!(lhs, from_coefficients(1, 7));
    }

    #[test]
    fn test_add_kb4() {
        let lhs = from_coefficients(1, 2);
        let rhs = kb4::Scalar::from_coefficients(3, 4, 5, 6);
        let expected = kb4::Scalar::from_coefficients(3, 4, 6, 8);
        assert_eq!(lhs + rhs, expected);
        assert_eq!(lhs + &rhs, expected);
    }

    #[test]
    fn test_add_kb8() {
        let lhs = from_coefficients(1, 2);
        let rhs = kb8::Scalar::from_coefficients(3, 4, 5, 6, 7, 8, 9, 10);
        let expected = kb8::Scalar::from_coefficients(3, 4, 5, 6, 7, 8, 10, 12);
        assert_eq!(lhs + rhs, expected);
        assert_eq!(lhs + &rhs, expected);
    }

    #[test]
    fn test_neg() {
        assert_eq!(-Scalar::ZERO, Scalar::ZERO);
        assert_eq!(
            -from_coefficients(1, 2),
            from_coefficients(MODULUS - 1, MODULUS - 2)
        );
        assert_eq!(
            from_coefficients(1, 2) + -from_coefficients(1, 2),
            Scalar::ZERO
        );
    }

    #[test]
    fn test_sub() {
        let lhs = from_coefficients(15, 27);
        let rhs = from_coefficients(5, 7);
        assert_eq!(lhs - rhs, from_coefficients(10, 20));
        assert_eq!(lhs - &rhs, from_coefficients(10, 20));
    }

    #[test]
    fn test_sub_wraparound() {
        let lhs = from_coefficients(5, 7);
        let rhs = from_coefficients(10, 10);
        assert_eq!(lhs - rhs, from_coefficients(MODULUS - 5, MODULUS - 3));
    }

    #[test]
    fn test_sub_assign() {
        let mut lhs = from_coefficients(15, 27);
        lhs -= from_coefficients(5, 7);
        assert_eq!(lhs, from_coefficients(10, 20));
    }

    #[test]
    fn test_sub_assign_ref() {
        let mut lhs = from_coefficients(15, 27);
        lhs -= &from_coefficients(5, 7);
        assert_eq!(lhs, from_coefficients(10, 20));
    }

    #[test]
    fn test_sub_base_scalar() {
        let lhs = from_coefficients(1, 7);
        let rhs = base::Scalar::from_const(5);
        assert_eq!(lhs - rhs, from_coefficients(1, 2));
        assert_eq!(lhs - &rhs, from_coefficients(1, 2));
    }

    #[test]
    fn test_sub_assign_base_scalar() {
        let rhs = base::Scalar::from_const(5);

        let mut lhs = from_coefficients(1, 7);
        lhs -= rhs;
        assert_eq!(lhs, from_coefficients(1, 2));

        let mut lhs = from_coefficients(1, 7);
        lhs -= &rhs;
        assert_eq!(lhs, from_coefficients(1, 2));
    }

    #[test]
    fn test_sub_kb4() {
        let lhs = from_coefficients(1, 2);
        let rhs = kb4::Scalar::from_coefficients(3, 4, 5, 6);
        let expected =
            kb4::Scalar::from_coefficients(MODULUS - 3, MODULUS - 4, MODULUS - 4, MODULUS - 4);
        assert_eq!(lhs - rhs, expected);
        assert_eq!(lhs - &rhs, expected);
    }

    #[test]
    fn test_sub_kb8() {
        let lhs = from_coefficients(1, 2);
        let rhs = kb8::Scalar::from_coefficients(3, 4, 5, 6, 7, 8, 9, 10);
        let expected = kb8::Scalar::from_coefficients(
            MODULUS - 3,
            MODULUS - 4,
            MODULUS - 5,
            MODULUS - 6,
            MODULUS - 7,
            MODULUS - 8,
            MODULUS - 8,
            MODULUS - 8,
        );
        assert_eq!(lhs - rhs, expected);
        assert_eq!(lhs - &rhs, expected);
    }

    #[test]
    fn test_extension_root() {
        let x = from_coefficients(1, 0);
        assert_eq!(x * x, from_const(QUADRATIC_NON_RESIDUE as u64));
        assert_eq!(x * &x, from_const(QUADRATIC_NON_RESIDUE as u64));
    }

    #[test]
    fn test_mul_by_zero() {
        assert_eq!(Scalar::ZERO * from_coefficients(2, 3), Scalar::ZERO);
        assert_eq!(from_coefficients(2, 3) * Scalar::ZERO, Scalar::ZERO);
    }

    #[test]
    fn test_mul_by_one() {
        assert_eq!(
            Scalar::ONE * from_coefficients(2, 3),
            from_coefficients(2, 3)
        );
        assert_eq!(
            from_coefficients(2, 3) * Scalar::ONE,
            from_coefficients(2, 3)
        );
    }

    #[test]
    fn test_mul() {
        let lhs = from_coefficients(2, 3);
        let rhs = from_coefficients(5, 7);
        let expected = from_coefficients(29, 21 + 10 * QUADRATIC_NON_RESIDUE);
        assert_eq!(lhs * rhs, expected);
        assert_eq!(lhs * &rhs, expected);
        assert_eq!(rhs * lhs, expected);
    }

    #[test]
    fn test_mul_assign() {
        let mut lhs = from_coefficients(2, 3);
        lhs *= from_coefficients(5, 7);
        assert_eq!(lhs, from_coefficients(29, 21 + 10 * QUADRATIC_NON_RESIDUE));
    }

    #[test]
    fn test_mul_base_scalar() {
        let lhs = from_coefficients(2, 3);
        let rhs = base::Scalar::from_const(5);
        assert_eq!(lhs * rhs, from_coefficients(10, 15));
        assert_eq!(lhs * &rhs, from_coefficients(10, 15));
    }

    #[test]
    fn test_mul_assign_base_scalar() {
        let rhs = base::Scalar::from_const(5);

        let mut lhs = from_coefficients(2, 3);
        lhs *= rhs;
        assert_eq!(lhs, from_coefficients(10, 15));

        let mut lhs = from_coefficients(2, 3);
        lhs *= &rhs;
        assert_eq!(lhs, from_coefficients(10, 15));
    }

    #[test]
    fn test_mul_kb4() {
        let lhs = from_coefficients(5, 6);
        let rhs = kb4::Scalar::from_coefficients(1, 2, 3, 4);
        let expected = kb4::Scalar::from_coefficients(16, 27, 38, 69);
        assert_eq!(lhs * rhs, expected);
        assert_eq!(lhs * &rhs, expected);
    }

    #[test]
    fn test_mul_kb8() {
        let lhs = from_coefficients(5, 6);
        let rhs = kb8::Scalar::from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);
        let expected = kb8::Scalar::from_coefficients(16, 27, 38, 69, 60, 111, 82, 153);
        assert_eq!(lhs * rhs, expected);
        assert_eq!(lhs * &rhs, expected);
    }

    #[test]
    fn test_div_by_one() {
        assert_eq!(
            from_coefficients(2, 3) / Scalar::ONE,
            from_coefficients(2, 3)
        );
        assert_eq!(
            from_coefficients(2, 3) / &Scalar::ONE,
            from_coefficients(2, 3)
        );
    }

    #[test]
    fn test_div() {
        let lhs = from_coefficients(2, 3);
        let rhs = from_coefficients(5, 7);
        assert_eq!((lhs * rhs) / rhs, lhs);
        assert_eq!((lhs * rhs) / &rhs, lhs);
    }

    #[test]
    fn test_div_base_scalar() {
        let lhs = from_coefficients(10, 15);
        let rhs = base::Scalar::from_const(5);
        assert_eq!(lhs / rhs, from_coefficients(2, 3));
        assert_eq!(lhs / &rhs, from_coefficients(2, 3));
    }

    #[test]
    fn test_div_assign_base_scalar() {
        let rhs = base::Scalar::from_const(5);

        let mut lhs = from_coefficients(10, 15);
        lhs /= rhs;
        assert_eq!(lhs, from_coefficients(2, 3));

        let mut lhs = from_coefficients(10, 15);
        lhs /= &rhs;
        assert_eq!(lhs, from_coefficients(2, 3));
    }

    #[test]
    fn test_div_kb4() {
        let lhs = from_coefficients(5, 6);
        let divisor = kb4::Scalar::from_coefficients(1, 2, 3, 4);
        assert_eq!((lhs / divisor) * divisor, kb4::Scalar::from(lhs));
        assert_eq!((lhs / &divisor) * divisor, kb4::Scalar::from(lhs));
    }

    #[test]
    fn test_div_kb8() {
        let lhs = from_coefficients(5, 6);
        let divisor = kb8::Scalar::from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);
        assert_eq!((lhs / divisor) * divisor, kb8::Scalar::from(lhs));
        assert_eq!((lhs / &divisor) * divisor, kb8::Scalar::from(lhs));
    }

    #[test]
    fn test_sum() {
        let values = vec![
            Scalar::ONE,
            from_coefficients(0, 2),
            from_coefficients(1, 0),
        ];
        assert_eq!(values.iter().sum::<Scalar>(), from_coefficients(1, 3));
        assert_eq!(values.into_iter().sum::<Scalar>(), from_coefficients(1, 3));
    }

    #[test]
    fn test_product() {
        let values = vec![
            from_coefficients(0, 2),
            from_coefficients(0, 3),
            from_coefficients(0, 4),
        ];
        assert_eq!(values.iter().product::<Scalar>(), from_coefficients(0, 24));
        assert_eq!(
            values.into_iter().product::<Scalar>(),
            from_coefficients(0, 24)
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
    }

    #[test]
    fn test_from_base_scalar() {
        assert_eq!(
            Scalar::from(base::Scalar::from_const(42)),
            from_coefficients(0, 42)
        );
        assert_eq!(Scalar::from(base::Scalar::ZERO), Scalar::ZERO);
    }

    #[test]
    fn test_try_from_usize() {
        assert_eq!(Scalar::try_from(0usize).unwrap(), from_const(0));
        assert_eq!(Scalar::try_from(42usize).unwrap(), from_const(42));
    }

    #[test]
    fn test_try_from_u64() {
        assert_eq!(Scalar::try_from(0u64).unwrap(), from_const(0));
        assert_eq!(Scalar::try_from(42u64).unwrap(), from_const(42));
        assert_eq!(Scalar::try_from(Scalar::MAX.to_u64()).unwrap(), Scalar::MAX);
        let modulus_squared = (MODULUS as u64) * (MODULUS as u64);
        assert!(Scalar::try_from(modulus_squared).is_err());
        assert!(Scalar::try_from(u64::MAX).is_err());
    }

    #[test]
    fn test_try_from_u128() {
        assert_eq!(Scalar::try_from(0u128).unwrap(), from_const(0));
        assert_eq!(Scalar::try_from(42u128).unwrap(), from_const(42));
        assert_eq!(
            Scalar::try_from(Scalar::MAX.to_u128()).unwrap(),
            Scalar::MAX
        );
        let modulus_squared = (MODULUS as u128) * (MODULUS as u128);
        assert!(Scalar::try_from(modulus_squared).is_err());
        assert!(Scalar::try_from(u64::MAX as u128).is_err());
        assert!(Scalar::try_from(u64::MAX as u128 + 1).is_err());
        assert!(Scalar::try_from(u128::MAX).is_err());
    }

    #[test]
    fn test_try_from_u256() {
        assert_eq!(Scalar::try_from(U256::from(0)).unwrap(), from_const(0));
        assert_eq!(Scalar::try_from(U256::from(42)).unwrap(), from_const(42));
        assert_eq!(
            Scalar::try_from(U256::from(Scalar::MAX.to_u64())).unwrap(),
            Scalar::MAX
        );
        assert!(Scalar::try_from(U256::from(u64::MAX)).is_err());
        assert!(Scalar::try_from(U256::from(u64::MAX) + U256::from(1)).is_err());
        assert!(Scalar::try_from(U256::MAX).is_err());
    }

    #[test]
    fn test_is_even() {
        assert!(bool::from(from_coefficients(0, 0).is_even()));
        assert!(bool::from(from_coefficients(1, 1).is_even()));
        assert!(!bool::from(from_coefficients(0, 1).is_even()));
        assert!(!bool::from(from_coefficients(1, 0).is_even()));
    }

    #[test]
    fn test_is_odd() {
        assert!(!bool::from(from_coefficients(0, 0).is_odd()));
        assert!(!bool::from(from_coefficients(1, 1).is_odd()));
        assert!(bool::from(from_coefficients(0, 1).is_odd()));
        assert!(bool::from(from_coefficients(1, 0).is_odd()));
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
        assert_eq!(from_coefficients(2, 3).double(), from_coefficients(4, 6));
    }

    #[test]
    fn test_square() {
        assert_eq!(from_coefficients(0, 5).square(), from_coefficients(0, 25));
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
        test_inversion_impl(from_coefficients(0, 42));
        test_inversion_impl(from_coefficients(1, 0));
        test_inversion_impl(from_coefficients(7, 11));
        test_inversion_impl(Scalar::MAX);
    }

    #[test]
    fn test_invert_batch() {
        let values = vec![
            from_coefficients(1, 2),
            from_coefficients(0, 42),
            Scalar::ONE,
            from_coefficients(3, 5),
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
    fn test_power() {
        let value = from_coefficients(1, 2);
        assert_eq!(value.pow(Scalar::ZERO), Scalar::ONE);
        assert_eq!(value.pow(Scalar::ONE), value);
        assert_eq!(value.pow(from_const(2)), value * value);
        assert_eq!(value.pow(from_const(3)), value * value * value);
    }

    #[test]
    fn test_power_vartime() {
        let value = from_coefficients(1, 2);
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
    fn test_try_from_le_bytes() {
        assert_eq!(
            Scalar::try_from_le_bytes(&42u64.to_le_bytes()).unwrap(),
            from_const(42)
        );
        assert!(bool::from(Scalar::try_from_le_bytes(&[255u8; 8]).is_none()));
    }

    #[test]
    fn test_try_from_be_bytes() {
        assert_eq!(
            Scalar::try_from_be_bytes(&42u64.to_be_bytes()).unwrap(),
            from_const(42)
        );
        assert!(bool::from(Scalar::try_from_be_bytes(&[255u8; 8]).is_none()));
    }

    #[test]
    fn test_le_be_bytes_roundtrip() {
        let value = from_coefficients(1, 42);
        let n = (MODULUS as u64) + 42;
        assert_eq!(Scalar::try_from_le_bytes(&n.to_le_bytes()).unwrap(), value);
        assert_eq!(Scalar::try_from_be_bytes(&n.to_be_bytes()).unwrap(), value);
    }

    #[test]
    fn test_fmt_display() {
        assert_eq!(format!("{}", from_const(0)), "0x0000000000000000");
        assert_eq!(format!("{}", from_const(0x13371337)), "0x0000000013371337");
        assert_eq!(format!("{}", Scalar::MAX), "0x3f010000fe000000");
    }

    #[test]
    fn test_fmt_debug() {
        assert_eq!(format!("{:?}", from_const(0)), "Scalar(0x0000000000000000)");
    }

    #[test]
    fn test_fmt_lower_hex() {
        assert_eq!(format!("{:x}", from_const(0x13371337)), "13371337");
        assert_eq!(format!("{:#x}", from_const(0x13371337)), "0x13371337");
        assert_eq!(format!("{:x}", Scalar::MAX), "3f010000fe000000");
    }

    #[test]
    fn test_fmt_upper_hex() {
        assert_eq!(format!("{:X}", from_const(0x13371337)), "13371337");
        assert_eq!(format!("{:X}", Scalar::MAX), "3F010000FE000000");
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
        assert_eq!("0x3f010000fe000000".parse::<Scalar>().unwrap(), Scalar::MAX);
        assert!("0x3f010000fe000001".parse::<Scalar>().is_err());
    }

    #[test]
    fn test_parse_scalar() {
        assert_eq!(parse_scalar("0x2a"), from_const(42));
    }

    #[test]
    fn test_try_to_u8() {
        assert_eq!(from_const(0).try_to_u8().unwrap(), 0);
        assert_eq!(from_const(u8::MAX as u64).try_to_u8().unwrap(), u8::MAX);
        assert!(from_const(u8::MAX as u64 + 1).try_to_u8().is_none());
        assert!(from_coefficients(1, 0).try_to_u8().is_none());
    }

    #[test]
    fn test_try_to_u16() {
        assert_eq!(from_const(0).try_to_u16().unwrap(), 0);
        assert_eq!(from_const(u16::MAX as u64).try_to_u16().unwrap(), u16::MAX);
        assert!(from_const(u16::MAX as u64 + 1).try_to_u16().is_none());
        assert!(from_coefficients(1, 0).try_to_u16().is_none());
    }

    #[test]
    fn test_field64_to_le_bytes() {
        let value = from_coefficients(1, 42);
        let n = (MODULUS as u64) + 42;
        assert_eq!(value.to_le_bytes(), n.to_le_bytes());
    }

    #[test]
    fn test_field64_to_be_bytes() {
        let value = from_coefficients(1, 42);
        let n = (MODULUS as u64) + 42;
        assert_eq!(value.to_be_bytes(), n.to_be_bytes());
    }

    #[test]
    fn test_field64_le_be_bytes_roundtrip() {
        let value = from_coefficients(7, 11);
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
    fn test_from_u128_mod_n() {
        assert_eq!(Scalar::from_u128_mod_n(0), from_const(0));
        assert_eq!(Scalar::from_u128_mod_n(42), from_const(42));
        assert_eq!(
            Scalar::from_u128_mod_n(MODULUS as u128),
            from_coefficients(1, 0)
        );
        let modulus_squared = (MODULUS as u128) * (MODULUS as u128);
        assert_eq!(Scalar::from_u128_mod_n(modulus_squared), Scalar::ZERO);
        assert_eq!(Scalar::from_u128_mod_n(modulus_squared + 1), Scalar::ONE);
        assert_eq!(
            Scalar::from_u128_mod_n((1u128 << 127) + 12345),
            from_const(0x381cbfd1033aa58e)
        );
    }

    #[test]
    fn test_from_u256_mod_n() {
        assert_eq!(Scalar::from_u256_mod_n(U256::from(0)), from_const(0));
        assert_eq!(Scalar::from_u256_mod_n(U256::from(42)), from_const(42));
        assert_eq!(
            Scalar::from_u256_mod_n(U256::from(MODULUS)),
            from_coefficients(1, 0)
        );
        let modulus_squared = U256::from(MODULUS) * U256::from(MODULUS);
        assert_eq!(Scalar::from_u256_mod_n(modulus_squared), Scalar::ZERO);
        assert_eq!(
            Scalar::from_u256_mod_n(modulus_squared + U256::from(1)),
            Scalar::ONE
        );
    }

    #[test]
    fn test_field64_try_to_u32() {
        assert_eq!(from_const(0).try_to_u32().unwrap(), 0);
        assert_eq!(from_const(42).try_to_u32().unwrap(), 42);
        assert_eq!(
            from_coefficients(0, MODULUS - 1).try_to_u32().unwrap(),
            MODULUS - 1
        );
        assert_eq!(from_coefficients(1, 0).try_to_u32().unwrap(), MODULUS);
        assert_eq!(
            from_coefficients(2, 33554429).try_to_u32().unwrap(),
            u32::MAX
        );
        assert!(bool::from(
            from_coefficients(2, 33554430).try_to_u32().is_none()
        ));
        assert!(bool::from(from_coefficients(3, 0).try_to_u32().is_none()));
    }

    #[test]
    fn test_to_u64() {
        assert_eq!(from_const(0).to_u64(), 0);
        assert_eq!(from_const(42).to_u64(), 42);
        assert_eq!(from_coefficients(1, 42).to_u64(), (MODULUS as u64) + 42);
        assert_eq!(
            Scalar::MAX.to_u64(),
            (MODULUS as u64) * (MODULUS as u64) - 1
        );
    }

    #[test]
    fn test_field64_to_u128() {
        assert_eq!(from_const(0).to_u128(), 0);
        assert_eq!(from_const(42).to_u128(), 42);
        assert_eq!(
            Scalar::MAX.to_u128(),
            (MODULUS as u128) * (MODULUS as u128) - 1
        );
    }

    #[test]
    fn test_field64_to_u256() {
        assert_eq!(from_const(0).to_u256(), U256::from(0));
        assert_eq!(from_const(42).to_u256(), U256::from(42));
        assert_eq!(Scalar::MAX.to_u256(), U256::from(Scalar::MAX.to_u64()));
    }

    #[test]
    fn test_field64_to_u512() {
        assert_eq!(from_const(0).to_u512(), U512::from(0));
        assert_eq!(from_const(42).to_u512(), U512::from(42));
        assert_eq!(Scalar::MAX.to_u512(), U512::from(Scalar::MAX.to_u64()));
    }
}
