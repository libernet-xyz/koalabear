use crate::helpers::{
    CHARACTERS_LOWER_CASE, CHARACTERS_UPPER_CASE, MODULUS, kb_add, kb_from_montgomery, kb_mul,
    kb_mul2x1, kb_mul4x1, kb_mul8x1, kb_sub, kb_to_montgomery,
};
use crate::kb2;
use crate::kb4;
use crate::kb8;
use anyhow::{Context, anyhow};
use primitive_types::{U256, U512};
use rand_core::{CryptoRng, TryCryptoRng};
use starkom_ff::{Field, Field32, PrimeField, PrimeField32};
use std::cmp::Ordering;
use std::fmt::{Binary, Debug, Display, Formatter, LowerHex, Octal, UpperHex};
use std::iter::{Product, Sum};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use std::str::FromStr;
use subtle::{
    Choice, ConditionallySelectable, ConstantTimeEq, ConstantTimeGreater, ConstantTimeLess,
    CtOption,
};

/// A KoalaBear scalar.
///
/// KoalaBear is a very fast 31-bit prime field with order `0x7f000001`.
///
/// Scalars are stored in Montgomery form with `R = 2^32`, so the inner value of a scalar `x` is
/// `x * 2^32 mod MODULUS` rather than `x` itself.
#[derive(Default, Copy, Clone, PartialEq, Eq)]
pub struct Scalar(pub(crate) u32);

impl Scalar {
    #[inline]
    const fn from_raw(value: u32) -> Self {
        assert!(value < MODULUS, "invalid KoalaBear value");
        Self(kb_to_montgomery(value))
    }

    /// Returns the raw (non-Montgomery) value of the scalar.
    #[inline]
    const fn to_raw(&self) -> u32 {
        kb_from_montgomery(self.0)
    }

    /// Constructs a KoalaBear scalar from its raw 32-bit value.
    ///
    /// Panics if the specified `value` exceeds [`MODULUS`].
    #[inline]
    pub const fn from_const(value: u32) -> Self {
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
        ((self.0 == other.0) as u8).into()
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
        Self(kb_add(self.0, rhs.0))
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
        self.0 = kb_add(self.0, rhs.0);
    }
}

impl<'a> AddAssign<&'a Self> for Scalar {
    fn add_assign(&mut self, rhs: &'a Self) {
        self.add_assign(*rhs);
    }
}

impl Add<kb2::Scalar> for Scalar {
    type Output = kb2::Scalar;

    fn add(self, rhs: kb2::Scalar) -> Self::Output {
        kb2::Scalar(rhs.0, kb_add(self.0, rhs.1))
    }
}

impl<'a> Add<&'a kb2::Scalar> for Scalar {
    type Output = kb2::Scalar;

    fn add(self, rhs: &'a kb2::Scalar) -> Self::Output {
        self.add(*rhs)
    }
}

impl Add<kb4::Scalar> for Scalar {
    type Output = kb4::Scalar;

    fn add(self, rhs: kb4::Scalar) -> Self::Output {
        kb4::Scalar(rhs.0, rhs.1, rhs.2, kb_add(self.0, rhs.3))
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
            rhs.6,
            kb_add(self.0, rhs.7),
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
        if self.0 > 0 {
            Self(MODULUS - self.0)
        } else {
            Self::ZERO
        }
    }
}

impl Sub<Self> for Scalar {
    type Output = Scalar;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(kb_sub(self.0, rhs.0))
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
        self.0 = kb_sub(self.0, rhs.0);
    }
}

impl<'a> SubAssign<&'a Self> for Scalar {
    fn sub_assign(&mut self, rhs: &'a Self) {
        self.sub_assign(*rhs);
    }
}

impl Sub<kb2::Scalar> for Scalar {
    type Output = kb2::Scalar;

    fn sub(self, rhs: kb2::Scalar) -> Self::Output {
        kb2::Scalar(kb_sub(0, rhs.0), kb_sub(self.0, rhs.1))
    }
}

impl<'a> Sub<&'a kb2::Scalar> for Scalar {
    type Output = kb2::Scalar;

    fn sub(self, rhs: &'a kb2::Scalar) -> Self::Output {
        self.sub(*rhs)
    }
}

impl Sub<kb4::Scalar> for Scalar {
    type Output = kb4::Scalar;

    fn sub(self, rhs: kb4::Scalar) -> Self::Output {
        kb4::Scalar(
            kb_sub(0, rhs.0),
            kb_sub(0, rhs.1),
            kb_sub(0, rhs.2),
            kb_sub(self.0, rhs.3),
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
            kb_sub(0, rhs.6),
            kb_sub(self.0, rhs.7),
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
        Self(kb_mul(self.0, rhs.0))
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
        self.0 = kb_mul(self.0, rhs.0);
    }
}

impl<'a> MulAssign<&'a Self> for Scalar {
    fn mul_assign(&mut self, rhs: &'a Self) {
        self.mul_assign(*rhs);
    }
}

impl Mul<kb2::Scalar> for Scalar {
    type Output = kb2::Scalar;

    fn mul(self, rhs: kb2::Scalar) -> Self::Output {
        let [c0, c1] = kb_mul2x1([rhs.0, rhs.1], self.0);
        kb2::Scalar(c0, c1)
    }
}

impl<'a> Mul<&'a kb2::Scalar> for Scalar {
    type Output = kb2::Scalar;

    fn mul(self, rhs: &'a kb2::Scalar) -> Self::Output {
        self.mul(*rhs)
    }
}

impl Mul<kb4::Scalar> for Scalar {
    type Output = kb4::Scalar;

    fn mul(self, rhs: kb4::Scalar) -> Self::Output {
        let [c0, c1, c2, c3] = kb_mul4x1([rhs.0, rhs.1, rhs.2, rhs.3], self.0);
        kb4::Scalar(c0, c1, c2, c3)
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
        let [c0, c1, c2, c3, c4, c5, c6, c7] = kb_mul8x1(
            [rhs.0, rhs.1, rhs.2, rhs.3, rhs.4, rhs.5, rhs.6, rhs.7],
            self.0,
        );
        kb8::Scalar(c0, c1, c2, c3, c4, c5, c6, c7)
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
        Self(kb_mul(self.0, rhs.invert_unwrap().0))
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
        self.0 = kb_mul(self.0, rhs.invert_unwrap().0);
    }
}

impl<'a> DivAssign<&'a Self> for Scalar {
    fn div_assign(&mut self, rhs: &'a Self) {
        self.div_assign(*rhs);
    }
}

impl Div<kb2::Scalar> for Scalar {
    type Output = kb2::Scalar;

    fn div(self, rhs: kb2::Scalar) -> Self::Output {
        self * rhs.invert_unwrap()
    }
}

impl<'a> Div<&'a kb2::Scalar> for Scalar {
    type Output = kb2::Scalar;

    fn div(self, rhs: &'a kb2::Scalar) -> Self::Output {
        self.div(*rhs)
    }
}

impl Div<kb4::Scalar> for Scalar {
    type Output = kb4::Scalar;

    fn div(self, rhs: kb4::Scalar) -> Self::Output {
        self * rhs.invert_unwrap()
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
        write!(f, "Scalar({:#010x})", self)
    }
}

impl Display for Scalar {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#010x}", self)
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
        Self(kb_to_montgomery(value as u32))
    }
}

impl From<u16> for Scalar {
    fn from(value: u16) -> Self {
        Self(kb_to_montgomery(value as u32))
    }
}

impl TryFrom<u32> for Scalar {
    type Error = anyhow::Error;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        Self::try_from_le_bytes(&value.to_le_bytes())
            .into_option()
            .context("overflow")
    }
}

impl TryFrom<usize> for Scalar {
    type Error = anyhow::Error;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        Self::try_from(u32::try_from(value).context("overflow")?)
    }
}

impl TryFrom<u64> for Scalar {
    type Error = anyhow::Error;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        Self::try_from(u32::try_from(value).context("overflow")?)
    }
}

impl TryFrom<u128> for Scalar {
    type Error = anyhow::Error;

    fn try_from(value: u128) -> Result<Self, Self::Error> {
        Self::try_from(u32::try_from(value).context("overflow")?)
    }
}

impl TryFrom<U256> for Scalar {
    type Error = anyhow::Error;

    fn try_from(value: U256) -> Result<Self, Self::Error> {
        if value > U256::from(u32::MAX) {
            Err(anyhow!("overflow"))
        } else {
            Self::try_from(value.as_u32())
        }
    }
}

impl Field for Scalar {
    type BaseField = Self;

    const MODULUS: &'static str = "0x7f000001";

    const CHARACTERISTIC: &'static str = "0x7f000001";

    const LEN: usize = 4;

    const ZERO: Self = Self(0);

    const ONE: Self = Self::from_const(1);

    const MAX: Self = Self::from_const(MODULUS - 1);

    const S: usize = 24;

    const MULTIPLICATIVE_GENERATOR: Self = Self::from_const(3);

    const MINUS_TWO: Self = Self::from_const(MODULUS - 2);

    const TWO_INV: Self = Self::from_const(0x3f800001);

    const ROOT_OF_UNITY: Self = Self::from_const(0x6ac49f88);

    const ROOT_OF_UNITY_INV: Self = Self::from_const(0x6b1073c7);

    const DELTA: Self = Self::from_const(0x6cf8fce2);

    fn is_odd(&self) -> Choice {
        ((self.to_raw() & 1) as u8).into()
    }

    fn try_random<R: TryCryptoRng>(rng: &mut R) -> Result<Self, R::Error> {
        let mut bytes = [0u8; 32];
        rng.try_fill_bytes(&mut bytes)?;
        Ok(Self::from_u256_mod_n(U256::from_little_endian(&bytes)))
    }

    fn random<R: CryptoRng>(rng: &mut R) -> Self {
        let mut bytes = [0u8; 32];
        rng.fill_bytes(&mut bytes);
        Self::from_u256_mod_n(U256::from_little_endian(&bytes))
    }

    fn random_default() -> Self {
        let mut bytes = [0u8; 32];
        getrandom::fill(&mut bytes).unwrap();
        Self::from_u256_mod_n(U256::from_little_endian(&bytes))
    }

    fn invert(&self) -> CtOption<Self> {
        CtOption::new(self.pow(Scalar::MINUS_TWO), !self.is_zero())
    }

    fn invert_vartime(&self) -> Option<Self> {
        if self.is_zero().into() {
            None
        } else {
            Some(self.pow_vartime(Scalar::MINUS_TWO))
        }
    }

    fn pow(mut self, exp: Self) -> Self {
        let mut exp = exp.to_raw();
        let mut result = Self::ONE;
        for _ in 0..32 {
            let product = result * self;
            result = Scalar::conditional_select(&result, &product, ((exp & 1) as u8).into());
            exp >>= 1;
            self = self.square();
        }
        result
    }

    fn pow_vartime(mut self, exp: Self) -> Self {
        let mut exp = exp.to_raw();
        let mut result = Self::ONE;
        while exp != 0 {
            if (exp & 1) != 0 {
                result *= self;
            }
            exp >>= 1;
            self = self.square();
        }
        result
    }

    fn div_int(&self, rhs: &Self) -> (Self, Self) {
        let lhs = self.to_raw();
        let rhs = rhs.to_raw();
        (
            Self(kb_to_montgomery(lhs / rhs)),
            Self(kb_to_montgomery(lhs % rhs)),
        )
    }

    fn try_from_le_bytes(bytes: &[u8]) -> CtOption<Self> {
        let mut fixed_bytes = [0u8; 4];
        fixed_bytes.copy_from_slice(bytes);
        let value = u32::from_le_bytes(fixed_bytes);
        CtOption::new(
            Self(kb_to_montgomery(value)),
            ((value < MODULUS) as u8).into(),
        )
    }

    fn try_from_be_bytes(bytes: &[u8]) -> CtOption<Self> {
        let mut fixed_bytes = [0u8; 4];
        fixed_bytes.copy_from_slice(bytes);
        let value = u32::from_be_bytes(fixed_bytes);
        CtOption::new(
            Self(kb_to_montgomery(value)),
            ((value < MODULUS) as u8).into(),
        )
    }

    fn from_str_radix(s: &str, radix: usize) -> Result<Self, std::fmt::Error> {
        assert!(radix >= 2 && radix <= 36);
        if s.is_empty() {
            return Err(std::fmt::Error);
        }
        let mut value = 0u64;
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
        Scalar::try_from(value).map_err(|_| std::fmt::Error)
    }

    fn to_str_radix(&self, radix: usize, pad_to: usize, upper_case: bool) -> String {
        assert!(radix >= 2 && radix <= 36);
        let characters = if upper_case {
            CHARACTERS_UPPER_CASE
        } else {
            CHARACTERS_LOWER_CASE
        };
        let mut value = self.to_raw();
        let mut s = String::default();
        let radix = radix as u32;
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
        let value = self.to_raw();
        if value > u8::MAX as u32 {
            None
        } else {
            Some(value as u8)
        }
    }

    fn try_to_u16(&self) -> Option<u16> {
        let value = self.to_raw();
        if value > u16::MAX as u32 {
            None
        } else {
            Some(value as u16)
        }
    }

    fn to_u256(&self) -> U256 {
        U256::from(self.to_raw())
    }

    fn to_u512(&self) -> U512 {
        U512::from(self.to_raw())
    }
}

impl Field32 for Scalar {
    fn to_le_bytes(&self) -> [u8; 4] {
        self.to_raw().to_le_bytes()
    }

    fn to_be_bytes(&self) -> [u8; 4] {
        self.to_raw().to_be_bytes()
    }

    fn from_u128_mod_n(u128: u128) -> Self {
        Self(kb_to_montgomery((u128 % (MODULUS as u128)) as u32))
    }

    fn from_u256_mod_n(u256: U256) -> Self {
        let value = u256 % U256::from(MODULUS);
        Self(kb_to_montgomery(value.as_u32()))
    }

    fn to_u32(&self) -> u32 {
        self.to_raw()
    }

    fn to_u64(&self) -> u64 {
        self.to_raw() as u64
    }

    fn to_u128(&self) -> u128 {
        self.to_raw() as u128
    }
}

impl PrimeField for Scalar {
    const ALPHA: usize = 3;
}

impl PrimeField32 for Scalar {}

#[cfg(test)]
mod tests {
    use super::*;

    #[inline]
    const fn from_const(value: u32) -> Scalar {
        Scalar::from_const(value)
    }

    #[inline]
    fn parse_scalar(s: &'static str) -> Scalar {
        s.parse().unwrap()
    }

    #[test]
    fn test_from_const() {
        assert_eq!(from_const(0), Scalar::ZERO);
        assert_eq!(from_const(1), Scalar::ONE);
        assert_eq!(from_const(0).to_string(), "0x00000000");
        assert_eq!(from_const(1).to_string(), "0x00000001");
        assert_eq!(from_const(0x13371337).to_string(), "0x13371337");
    }

    #[test]
    #[should_panic(expected = "invalid KoalaBear value")]
    fn test_from_const_out_of_range() {
        from_const(MODULUS);
    }

    #[test]
    fn test_modulus() {
        assert_eq!(Scalar::MODULUS, "0x7f000001");
        assert_eq!(Scalar::MAX, from_const(MODULUS - 1));
    }

    #[test]
    fn test_zero() {
        assert_eq!(Scalar::ZERO, Scalar::zero());
        assert_eq!(Scalar::ZERO, from_const(0));
        assert_eq!(Scalar::ZERO + from_const(0), from_const(0));
        assert_eq!(Scalar::ZERO + from_const(1), from_const(1));
        assert_eq!(Scalar::ZERO * from_const(42), Scalar::ZERO);
    }

    #[test]
    fn test_one() {
        assert_eq!(Scalar::ONE, Scalar::one());
        assert_eq!(Scalar::ONE, from_const(1));
        assert_eq!(Scalar::ONE + from_const(0), from_const(1));
        assert_eq!(Scalar::ONE + from_const(1), from_const(2));
        assert_eq!(Scalar::ONE * from_const(42), from_const(42));
    }

    #[test]
    fn test_max() {
        assert_eq!(Scalar::MAX, -Scalar::ONE);
        assert_eq!(Scalar::MAX, from_const(0x7f000000));
    }

    #[test]
    fn test_fmt_display() {
        assert_eq!(format!("{}", from_const(0)), "0x00000000");
        assert_eq!(format!("{}", from_const(1)), "0x00000001");
        assert_eq!(format!("{}", from_const(0x13371337)), "0x13371337");
        assert_eq!(format!("{}", Scalar::MAX - Scalar::ONE), "0x7effffff");
        assert_eq!(format!("{}", Scalar::MAX), "0x7f000000");
    }

    #[test]
    fn test_fmt_debug() {
        assert_eq!(format!("{:?}", from_const(0)), "Scalar(0x00000000)");
        assert_eq!(format!("{:?}", from_const(1)), "Scalar(0x00000001)");
        assert_eq!(format!("{:?}", Scalar::MAX), "Scalar(0x7f000000)");
    }

    #[test]
    fn test_fmt_lower_hex() {
        assert_eq!(format!("{:x}", from_const(0)), "0");
        assert_eq!(format!("{:x}", from_const(1)), "1");
        assert_eq!(format!("{:x}", from_const(0x13371337)), "13371337");
        assert_eq!(format!("{:x}", Scalar::MAX), "7f000000");
        assert_eq!(format!("{:#x}", from_const(0)), "0x0");
        assert_eq!(format!("{:#x}", from_const(0x13371337)), "0x13371337");
        assert_eq!(format!("{:10x}", from_const(0x13371337)), "  13371337");
        assert_eq!(format!("{:010x}", from_const(0x13371337)), "0013371337");
        assert_eq!(format!("{:#012x}", from_const(0x13371337)), "0x0013371337");
        assert_eq!(format!("{:<10x}", from_const(0x13371337)), "13371337  ");
    }

    #[test]
    fn test_fmt_upper_hex() {
        assert_eq!(format!("{:X}", from_const(0)), "0");
        assert_eq!(format!("{:X}", from_const(0x13371337)), "13371337");
        assert_eq!(format!("{:X}", Scalar::MAX), "7F000000");
        assert_eq!(format!("{:#X}", from_const(0x13371337)), "0x13371337");
        assert_eq!(format!("{:#X}", Scalar::MAX), "0x7F000000");
        assert_eq!(format!("{:010X}", from_const(0x13371337)), "0013371337");
    }

    #[test]
    fn test_fmt_binary() {
        assert_eq!(format!("{:b}", from_const(0)), "0");
        assert_eq!(format!("{:b}", from_const(0b1010)), "1010");
        assert_eq!(format!("{:#b}", from_const(0b1010)), "0b1010");
        assert_eq!(format!("{:010b}", from_const(0b1010)), "0000001010");
    }

    #[test]
    fn test_fmt_octal() {
        assert_eq!(format!("{:o}", from_const(0)), "0");
        assert_eq!(format!("{:o}", from_const(0o755)), "755");
        assert_eq!(format!("{:#o}", from_const(0o755)), "0o755");
        assert_eq!(format!("{:010o}", from_const(0o755)), "0000000755");
    }

    #[test]
    fn test_equality() {
        assert!(from_const(0) == from_const(0));
        assert!(from_const(0) != from_const(1));
        assert!(from_const(1) == from_const(1));
        assert!(Scalar::MAX != Scalar::MAX - Scalar::ONE);
        assert!(Scalar::MAX == Scalar::MAX);
    }

    #[test]
    fn test_total_order() {
        let v0 = from_const(0);
        let v1 = from_const(1);
        let v2 = from_const(42);
        let v3 = from_const(0x01234567);
        let v4 = Scalar::MAX - Scalar::ONE;
        let v5 = Scalar::MAX;

        assert_eq!(v0.cmp(&v0), Ordering::Equal);
        assert_eq!(v0.cmp(&v1), Ordering::Less);
        assert_eq!(v1.cmp(&v0), Ordering::Greater);
        assert_eq!(v1.cmp(&v2), Ordering::Less);
        assert_eq!(v2.cmp(&v3), Ordering::Less);
        assert_eq!(v3.cmp(&v4), Ordering::Less);
        assert_eq!(v4.cmp(&v5), Ordering::Less);
        assert_eq!(v5.cmp(&v4), Ordering::Greater);
        assert_eq!(v5.cmp(&v5), Ordering::Equal);
    }

    #[test]
    fn test_ct_eq() {
        let a = from_const(42);
        let b = from_const(42);
        let c = from_const(43);
        assert_eq!(bool::from(a.ct_eq(&b)), true);
        assert_eq!(bool::from(a.ct_eq(&a)), true);
        assert_eq!(bool::from(a.ct_eq(&c)), false);
        assert_eq!(bool::from(Scalar::ZERO.ct_eq(&Scalar::ONE)), false);
        assert_eq!(bool::from(Scalar::MAX.ct_eq(&Scalar::MAX)), true);
    }

    #[test]
    fn test_ct_gt() {
        let v0 = from_const(0);
        let v1 = from_const(1);
        let v2 = from_const(42);
        let v4 = Scalar::MAX;
        assert_eq!(bool::from(v0.ct_gt(&v0)), false);
        assert_eq!(bool::from(v1.ct_gt(&v0)), true);
        assert_eq!(bool::from(v2.ct_gt(&v1)), true);
        assert_eq!(bool::from(v4.ct_gt(&v0)), true);
        assert_eq!(bool::from(v0.ct_gt(&v4)), false);
    }

    #[test]
    fn test_ct_lt() {
        let v0 = from_const(0);
        let v1 = from_const(1);
        let v2 = from_const(42);
        let v4 = Scalar::MAX;
        assert_eq!(bool::from(v0.ct_lt(&v0)), false);
        assert_eq!(bool::from(v0.ct_lt(&v1)), true);
        assert_eq!(bool::from(v1.ct_lt(&v2)), true);
        assert_eq!(bool::from(v0.ct_lt(&v4)), true);
        assert_eq!(bool::from(v4.ct_lt(&v0)), false);
    }

    #[test]
    fn test_conditional_select() {
        let a = from_const(12);
        let b = from_const(34);
        assert_eq!(Scalar::conditional_select(&a, &b, Choice::from(0)), a);
        assert_eq!(Scalar::conditional_select(&a, &b, Choice::from(1)), b);
        assert_eq!(
            Scalar::conditional_select(&Scalar::ZERO, &Scalar::ONE, Choice::from(0)),
            Scalar::ZERO
        );
        assert_eq!(
            Scalar::conditional_select(&Scalar::ZERO, &Scalar::ONE, Choice::from(1)),
            Scalar::ONE
        );
    }

    #[test]
    fn test_add() {
        let lhs = from_const(0x01234567);
        let rhs = from_const(0x0fedcba9);
        assert_eq!(lhs + rhs, from_const(0x11111110));
        assert_eq!(lhs + &rhs, from_const(0x11111110));
    }

    #[test]
    fn test_add_wraparound() {
        let lhs = from_const(MODULUS - 5);
        let rhs = from_const(10);
        assert_eq!(lhs + rhs, from_const(5));
        assert_eq!(lhs + &rhs, from_const(5));
    }

    #[test]
    fn test_add_assign() {
        let mut lhs = from_const(0x01234567);
        lhs += from_const(0x0fedcba9);
        assert_eq!(lhs, from_const(0x11111110));
    }

    #[test]
    fn test_add_assign_ref() {
        let mut lhs = from_const(0x01234567);
        lhs += &from_const(0x0fedcba9);
        assert_eq!(lhs, from_const(0x11111110));
    }

    #[test]
    fn test_add_assign_wraparound() {
        let mut lhs = from_const(MODULUS - 5);
        lhs += from_const(10);
        assert_eq!(lhs, from_const(5));
    }

    #[test]
    fn test_add_kb2() {
        let lhs = from_const(5);
        let rhs = kb2::Scalar::from_coefficients(0, 7);
        let expected = kb2::Scalar::from_coefficients(0, 12);
        assert_eq!(lhs + rhs, expected);
        assert_eq!(lhs + &rhs, expected);
    }

    #[test]
    fn test_add_kb4() {
        let lhs = from_const(5);
        let rhs = kb4::Scalar::from_coefficients(1, 2, 3, 7);
        let expected = kb4::Scalar::from_coefficients(1, 2, 3, 12);
        assert_eq!(lhs + rhs, expected);
        assert_eq!(lhs + &rhs, expected);
    }

    #[test]
    fn test_add_kb8() {
        let lhs = from_const(5);
        let rhs = kb8::Scalar::from_coefficients(1, 2, 3, 4, 5, 6, 7, 7);
        let expected = kb8::Scalar::from_coefficients(1, 2, 3, 4, 5, 6, 7, 12);
        assert_eq!(lhs + rhs, expected);
        assert_eq!(lhs + &rhs, expected);
    }

    fn test_neg_impl(value: Scalar) {
        assert_eq!(-value, Scalar::MAX - value + Scalar::ONE);
    }

    #[test]
    fn test_neg() {
        assert_eq!(-Scalar::ZERO, Scalar::ZERO);
        assert_eq!(-Scalar::ONE, Scalar::MAX);
        assert_eq!(-from_const(2), Scalar::MAX - Scalar::ONE);
        test_neg_impl(from_const(0x01234567));
        test_neg_impl(from_const(0x0fedcba9));
        test_neg_impl(Scalar::MAX);
    }

    #[test]
    fn test_sub() {
        let lhs = from_const(0x11111110);
        let rhs = from_const(0x0fedcba9);
        assert_eq!(lhs - rhs, from_const(0x01234567));
        assert_eq!(lhs - &rhs, from_const(0x01234567));
    }

    #[test]
    fn test_sub_wraparound() {
        let lhs = from_const(3);
        let rhs = from_const(10);
        assert_eq!(lhs - rhs, from_const(MODULUS - 7));
        assert_eq!(lhs - &rhs, from_const(MODULUS - 7));
    }

    #[test]
    fn test_sub_assign() {
        let mut lhs = from_const(0x11111110);
        lhs -= from_const(0x0fedcba9);
        assert_eq!(lhs, from_const(0x01234567));
    }

    #[test]
    fn test_sub_assign_ref() {
        let mut lhs = from_const(0x11111110);
        lhs -= &from_const(0x0fedcba9);
        assert_eq!(lhs, from_const(0x01234567));
    }

    #[test]
    fn test_sub_assign_wraparound() {
        let mut lhs = from_const(3);
        lhs -= from_const(10);
        assert_eq!(lhs, from_const(MODULUS - 7));
    }

    #[test]
    fn test_sub_kb2() {
        let lhs = from_const(12);
        let rhs = kb2::Scalar::from_coefficients(0, 7);
        let expected = kb2::Scalar::from_coefficients(0, 5);
        assert_eq!(lhs - rhs, expected);
        assert_eq!(lhs - &rhs, expected);
    }

    #[test]
    fn test_sub_kb4() {
        let lhs = from_const(5);
        let rhs = kb4::Scalar::from_coefficients(1, 2, 3, 7);
        let expected =
            kb4::Scalar::from_coefficients(MODULUS - 1, MODULUS - 2, MODULUS - 3, MODULUS - 2);
        assert_eq!(lhs - rhs, expected);
        assert_eq!(lhs - &rhs, expected);
    }

    #[test]
    fn test_sub_kb8() {
        let lhs = from_const(5);
        let rhs = kb8::Scalar::from_coefficients(1, 2, 3, 4, 5, 6, 7, 7);
        let expected = kb8::Scalar::from_coefficients(
            MODULUS - 1,
            MODULUS - 2,
            MODULUS - 3,
            MODULUS - 4,
            MODULUS - 5,
            MODULUS - 6,
            MODULUS - 7,
            MODULUS - 2,
        );
        assert_eq!(lhs - rhs, expected);
        assert_eq!(lhs - &rhs, expected);
    }

    #[test]
    fn test_mul_by_zero() {
        assert_eq!(Scalar::ZERO * from_const(42), Scalar::ZERO);
        assert_eq!(Scalar::ZERO * &from_const(42), Scalar::ZERO);
        assert_eq!(from_const(42) * Scalar::ZERO, Scalar::ZERO);
    }

    #[test]
    fn test_mul_by_one() {
        assert_eq!(Scalar::ONE * from_const(42), from_const(42));
        assert_eq!(from_const(42) * Scalar::ONE, from_const(42));
    }

    #[test]
    fn test_mul() {
        assert_eq!(from_const(12) * from_const(34), from_const(408));
        assert_eq!(from_const(12) * &from_const(34), from_const(408));
        assert_eq!(from_const(34) * from_const(12), from_const(408));
    }

    #[test]
    fn test_mul_large() {
        let v1 = from_const(0x76543210);
        let v2 = from_const(0x12345678);
        let v3 = from_const(0x7292b208);
        assert_eq!(v1 * v2, v3);
        assert_eq!(v1 * &v2, v3);
        assert_eq!(v2 * v1, v3);
        assert_eq!(v2 * &v1, v3);
    }

    #[test]
    fn test_mul_kb2() {
        let lhs = from_const(12);
        let rhs = kb2::Scalar::from_coefficients(0, 7);
        let expected = kb2::Scalar::from_coefficients(0, 84);
        assert_eq!(lhs * rhs, expected);
        assert_eq!(lhs * &rhs, expected);
    }

    #[test]
    fn test_mul_kb4() {
        let lhs = from_const(5);
        let rhs = kb4::Scalar::from_coefficients(1, 2, 3, 4);
        let expected = kb4::Scalar::from_coefficients(5, 10, 15, 20);
        assert_eq!(lhs * rhs, expected);
        assert_eq!(lhs * &rhs, expected);
    }

    #[test]
    fn test_mul_kb8() {
        let lhs = from_const(5);
        let rhs = kb8::Scalar::from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);
        let expected = kb8::Scalar::from_coefficients(5, 10, 15, 20, 25, 30, 35, 40);
        assert_eq!(lhs * rhs, expected);
        assert_eq!(lhs * &rhs, expected);
    }

    #[test]
    fn test_div_by_one() {
        assert_eq!(from_const(42) / Scalar::ONE, from_const(42));
        assert_eq!(from_const(42) / &Scalar::ONE, from_const(42));
    }

    #[test]
    fn test_div() {
        assert_eq!(from_const(408) / from_const(34), from_const(12));
        assert_eq!(from_const(408) / &from_const(34), from_const(12));
        assert_eq!(from_const(408) / from_const(12), from_const(34));
    }

    #[test]
    fn test_div_kb2() {
        let lhs = from_const(84);
        let rhs = kb2::Scalar::from_coefficients(0, 7);
        let expected = kb2::Scalar::from_coefficients(0, 12);
        assert_eq!(lhs / rhs, expected);
        assert_eq!(lhs / &rhs, expected);
    }

    #[test]
    fn test_div_kb4() {
        let lhs = from_const(5);
        let divisor = kb4::Scalar::from_coefficients(1, 2, 3, 4);
        assert_eq!((lhs / divisor) * divisor, kb4::Scalar::from(lhs));
        assert_eq!((lhs / &divisor) * divisor, kb4::Scalar::from(lhs));
    }

    #[test]
    fn test_div_kb8() {
        let lhs = from_const(5);
        let divisor = kb8::Scalar::from_coefficients(1, 2, 3, 4, 5, 6, 7, 8);
        assert_eq!((lhs / divisor) * divisor, kb8::Scalar::from(lhs));
        assert_eq!((lhs / &divisor) * divisor, kb8::Scalar::from(lhs));
    }

    #[test]
    fn test_sum_owned() {
        let values = vec![Scalar::ONE, from_const(2), from_const(3)];
        assert_eq!(values.into_iter().sum::<Scalar>(), from_const(6));
    }

    #[test]
    fn test_sum_refs() {
        let values = vec![Scalar::ONE, from_const(2), from_const(3)];
        assert_eq!(values.iter().sum::<Scalar>(), from_const(6));
    }

    #[test]
    fn test_sum_empty() {
        let values: Vec<Scalar> = vec![];
        assert_eq!(values.into_iter().sum::<Scalar>(), Scalar::ZERO);
    }

    #[test]
    fn test_sum_wraps_modulo_p() {
        let values = vec![Scalar::MAX, Scalar::ONE];
        assert_eq!(values.into_iter().sum::<Scalar>(), Scalar::ZERO);
    }

    #[test]
    fn test_product_owned() {
        let values = vec![from_const(2), from_const(3), from_const(4)];
        assert_eq!(values.into_iter().product::<Scalar>(), from_const(24));
    }

    #[test]
    fn test_product_refs() {
        let values = vec![from_const(2), from_const(3), from_const(4)];
        assert_eq!(values.iter().product::<Scalar>(), from_const(24));
    }

    #[test]
    fn test_product_empty() {
        let values: Vec<Scalar> = vec![];
        assert_eq!(values.into_iter().product::<Scalar>(), Scalar::ONE);
    }

    #[test]
    fn test_product_with_zero() {
        let values = vec![from_const(5), Scalar::ZERO, from_const(7)];
        assert_eq!(values.into_iter().product::<Scalar>(), Scalar::ZERO);
    }

    #[test]
    fn test_from_u8() {
        assert_eq!(Scalar::from(0u8), from_const(0));
        assert_eq!(Scalar::from(u8::MAX), from_const(u8::MAX as u32));
    }

    #[test]
    fn test_from_u16() {
        assert_eq!(Scalar::from(0u16), from_const(0));
        assert_eq!(Scalar::from(u16::MAX), from_const(u16::MAX as u32));
    }

    #[test]
    fn test_try_from_u32() {
        assert_eq!(Scalar::try_from(0u32).unwrap(), from_const(0));
        assert_eq!(Scalar::try_from(1u32).unwrap(), from_const(1));
        assert_eq!(
            Scalar::try_from(MODULUS - 1).unwrap(),
            from_const(MODULUS - 1)
        );
        assert!(Scalar::try_from(MODULUS).is_err());
        assert!(Scalar::try_from(u32::MAX).is_err());
    }

    #[test]
    fn test_try_from_usize() {
        assert_eq!(Scalar::try_from(0usize).unwrap(), from_const(0));
        assert_eq!(Scalar::try_from(1usize).unwrap(), from_const(1));
        assert_eq!(
            Scalar::try_from((MODULUS - 1) as usize).unwrap(),
            from_const(MODULUS - 1)
        );
        assert!(Scalar::try_from(MODULUS as usize).is_err());
        assert!(Scalar::try_from(usize::MAX).is_err());
    }

    #[test]
    fn test_try_from_u64() {
        assert_eq!(Scalar::try_from(0u64).unwrap(), from_const(0));
        assert_eq!(Scalar::try_from(1u64).unwrap(), from_const(1));
        assert_eq!(
            Scalar::try_from((MODULUS - 1) as u64).unwrap(),
            from_const(MODULUS - 1)
        );
        assert!(Scalar::try_from(MODULUS as u64).is_err());
        assert!(Scalar::try_from(u32::MAX as u64).is_err());
        assert!(Scalar::try_from(u32::MAX as u64 + 1).is_err());
        assert!(Scalar::try_from(u64::MAX).is_err());
    }

    #[test]
    fn test_try_from_u128() {
        assert_eq!(Scalar::try_from(0u128).unwrap(), from_const(0));
        assert_eq!(Scalar::try_from(1u128).unwrap(), from_const(1));
        assert_eq!(
            Scalar::try_from((MODULUS - 1) as u128).unwrap(),
            from_const(MODULUS - 1)
        );
        assert!(Scalar::try_from(MODULUS as u128).is_err());
        assert!(Scalar::try_from(u32::MAX as u128).is_err());
        assert!(Scalar::try_from(u32::MAX as u128 + 1).is_err());
        assert!(Scalar::try_from(u128::MAX).is_err());
    }

    #[test]
    fn test_try_from_u256() {
        assert_eq!(Scalar::try_from(U256::from(0)).unwrap(), from_const(0));
        assert_eq!(Scalar::try_from(U256::from(1)).unwrap(), from_const(1));
        assert_eq!(
            Scalar::try_from(U256::from(MODULUS - 1)).unwrap(),
            from_const(MODULUS - 1)
        );
        assert!(Scalar::try_from(U256::from(MODULUS)).is_err());
        assert!(Scalar::try_from(U256::from(u32::MAX)).is_err());
        assert!(Scalar::try_from(U256::from(u32::MAX) + U256::from(1)).is_err());
        assert!(Scalar::try_from(U256::MAX).is_err());
    }

    #[test]
    fn test_is_zero() {
        assert!(bool::from(from_const(0).is_zero()));
        assert!(!bool::from(from_const(1).is_zero()));
        assert!(!bool::from(Scalar::MAX.is_zero()));
    }

    #[test]
    fn test_is_even() {
        assert!(bool::from(from_const(0).is_even()));
        assert!(!bool::from(from_const(1).is_even()));
        assert!(bool::from(from_const(2).is_even()));
        assert!(bool::from(Scalar::MAX.is_even()));
        assert!(!bool::from((Scalar::MAX - Scalar::ONE).is_even()));
    }

    #[test]
    fn test_is_odd() {
        assert!(!bool::from(from_const(0).is_odd()));
        assert!(bool::from(from_const(1).is_odd()));
        assert!(!bool::from(from_const(2).is_odd()));
        assert!(!bool::from(Scalar::MAX.is_odd()));
        assert!(bool::from((Scalar::MAX - Scalar::ONE).is_odd()));
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
        assert_eq!(from_const(0).double(), from_const(0));
        assert_eq!(from_const(2).double(), from_const(4));
        assert_eq!((Scalar::MAX - from_const(2)).double(), -from_const(6));
        assert_eq!(Scalar::MAX.double(), -from_const(2));
    }

    #[test]
    fn test_square() {
        assert_eq!(from_const(0).square(), from_const(0));
        assert_eq!(from_const(2).square(), from_const(4));
        assert_eq!((Scalar::MAX - from_const(2)).square(), from_const(9));
        assert_eq!(Scalar::MAX.square(), from_const(1));
    }

    #[test]
    fn test_cube() {
        assert_eq!(from_const(0).cube(), from_const(0));
        assert_eq!(from_const(2).cube(), from_const(8));
        assert_eq!((Scalar::MAX - from_const(2)).cube(), -from_const(27));
        assert_eq!(Scalar::MAX.cube(), -from_const(1));
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
        assert!(from_const(0).invert_vartime().is_none());
        assert_eq!(from_const(0).invert_or_zero(), Scalar::ZERO);
        assert!(bool::from(from_const(0).invert().is_none()));
        test_inversion_impl(from_const(1));
        test_inversion_impl(from_const(2));
        test_inversion_impl(from_const(42));
        test_inversion_impl(Scalar::MAX);
    }

    fn test_invert_batch_impl(values: &[Scalar]) {
        let expected: Vec<Scalar> = values
            .iter()
            .map(|value| value.invert_vartime().unwrap())
            .collect();

        let mut batch = values.to_vec();
        Scalar::invert_batch(&mut batch);
        assert_eq!(batch, expected);

        batch = values.to_vec();
        Scalar::invert_batch_vartime(&mut batch);
        assert_eq!(batch, expected);
    }

    #[test]
    fn test_invert_batch() {
        test_invert_batch_impl(&[]);
        test_invert_batch_impl(&[Scalar::ONE]);
        test_invert_batch_impl(&[from_const(42)]);
        test_invert_batch_impl(&[Scalar::MAX]);
        test_invert_batch_impl(&[from_const(1), from_const(2), from_const(3)]);
        test_invert_batch_impl(&[
            from_const(42),
            Scalar::ONE,
            Scalar::MAX,
            from_const(1),
            from_const(2),
            Scalar::MAX - from_const(1),
        ]);
    }

    #[test]
    fn test_power() {
        assert_eq!(from_const(0).pow(from_const(0)), from_const(1));
        assert_eq!(from_const(0).pow(from_const(1)), from_const(0));
        assert_eq!(from_const(2).pow(from_const(0)), from_const(1));
        assert_eq!(from_const(2).pow(from_const(1)), from_const(2));
        assert_eq!(from_const(2).pow(from_const(2)), from_const(4));
        assert_eq!(from_const(2).pow(from_const(10)), from_const(1024));
    }

    #[test]
    fn test_small_power() {
        assert_eq!(from_const(0).pow_small(0), from_const(1));
        assert_eq!(from_const(2).pow_small(0), from_const(1));
        assert_eq!(from_const(2).pow_small(10), from_const(1024));
    }

    #[test]
    fn test_power_vartime() {
        assert_eq!(from_const(0).pow_vartime(from_const(0)), from_const(1));
        assert_eq!(from_const(2).pow_vartime(from_const(0)), from_const(1));
        assert_eq!(from_const(2).pow_vartime(from_const(10)), from_const(1024));
    }

    #[test]
    fn test_small_power_vartime() {
        assert_eq!(from_const(0).pow_small_vartime(0), from_const(1));
        assert_eq!(from_const(2).pow_small_vartime(0), from_const(1));
        assert_eq!(from_const(2).pow_small_vartime(10), from_const(1024));
    }

    #[test]
    fn test_integer_division() {
        assert_eq!(
            from_const(13).div_int(&from_const(5)),
            (from_const(2), from_const(3))
        );
        assert_eq!(
            from_const(61).div_int(&from_const(7)),
            (from_const(8), from_const(5))
        );
    }

    #[test]
    fn test_try_from_le_bytes() {
        assert_eq!(
            Scalar::try_from_le_bytes(&[103, 69, 35, 1]).unwrap(),
            from_const(0x01234567)
        );
        assert_eq!(
            Scalar::try_from_le_bytes(&[0, 0, 0, 127]).unwrap(),
            Scalar::MAX
        );
        assert!(bool::from(
            Scalar::try_from_le_bytes(&[1, 0, 0, 127]).is_none()
        ));
    }

    #[test]
    fn test_try_from_be_bytes() {
        assert_eq!(
            Scalar::try_from_be_bytes(&[1, 35, 69, 103]).unwrap(),
            from_const(0x01234567)
        );
        assert_eq!(
            Scalar::try_from_be_bytes(&[127, 0, 0, 0]).unwrap(),
            Scalar::MAX
        );
        assert!(bool::from(
            Scalar::try_from_be_bytes(&[127, 0, 0, 1]).is_none()
        ));
    }

    #[test]
    fn test_parse_binary() {
        assert_eq!(Scalar::from_str_radix("0", 2).unwrap(), from_const(0));
        assert_eq!(Scalar::from_str_radix("1", 2).unwrap(), from_const(1));
        assert_eq!(Scalar::from_str_radix("101010", 2).unwrap(), from_const(42));
        assert_eq!(
            Scalar::from_str_radix("1111110111111111111111111111111", 2).unwrap(),
            Scalar::MAX - Scalar::ONE
        );
        assert_eq!(
            Scalar::from_str_radix("1111111000000000000000000000000", 2).unwrap(),
            Scalar::MAX
        );
        assert!(Scalar::from_str_radix("1111111000000000000000000000001", 2).is_err());
    }

    #[test]
    fn test_print_binary() {
        assert_eq!(from_const(0).to_str_radix(2, 0, false), "0");
        assert_eq!(from_const(42).to_str_radix(2, 0, false), "101010");
        assert_eq!(from_const(42).to_str_radix(2, 10, false), "0000101010");
        assert_eq!(
            Scalar::MAX.to_str_radix(2, 0, false),
            "1111111000000000000000000000000"
        );
    }

    #[test]
    fn test_parse_octal() {
        assert_eq!(Scalar::from_str_radix("0", 8).unwrap(), from_const(0));
        assert_eq!(Scalar::from_str_radix("52", 8).unwrap(), from_const(42));
        assert!(Scalar::from_str_radix("8", 8).is_err());
        assert_eq!(
            Scalar::from_str_radix("17700000000", 8).unwrap(),
            Scalar::MAX
        );
        assert!(Scalar::from_str_radix("17700000001", 8).is_err());
        assert!(Scalar::from_str_radix("20000000000", 8).is_err());
    }

    #[test]
    fn test_print_octal() {
        assert_eq!(from_const(0).to_str_radix(8, 0, false), "0");
        assert_eq!(from_const(42).to_str_radix(8, 0, false), "52");
        assert_eq!(Scalar::MAX.to_str_radix(8, 0, false), "17700000000");
    }

    #[test]
    fn test_parse_decimal() {
        assert_eq!(Scalar::from_str_radix("0", 10).unwrap(), from_const(0));
        assert_eq!(Scalar::from_str_radix("42", 10).unwrap(), from_const(42));
        assert_eq!(
            Scalar::from_str_radix("2130706431", 10).unwrap(),
            Scalar::MAX - Scalar::ONE
        );
        assert_eq!(
            Scalar::from_str_radix("2130706432", 10).unwrap(),
            Scalar::MAX
        );
        assert!(Scalar::from_str_radix("2130706433", 10).is_err());
    }

    #[test]
    fn test_print_decimal() {
        assert_eq!(from_const(0).to_str_radix(10, 0, false), "0");
        assert_eq!(from_const(42).to_str_radix(10, 0, false), "42");
        assert_eq!(Scalar::MAX.to_str_radix(10, 0, false), "2130706432");
    }

    #[test]
    fn test_parse_hexadecimal_lower_case() {
        assert_eq!(Scalar::from_str_radix("0", 16).unwrap(), from_const(0));
        assert_eq!(Scalar::from_str_radix("2a", 16).unwrap(), from_const(42));
        assert_eq!(Scalar::from_str_radix("7f000000", 16).unwrap(), Scalar::MAX);
        assert!(Scalar::from_str_radix("7f000001", 16).is_err());
    }

    #[test]
    fn test_print_hexadecimal_lower_case() {
        assert_eq!(from_const(0).to_str_radix(16, 0, false), "0");
        assert_eq!(
            from_const(0x13371337).to_str_radix(16, 0, false),
            "13371337"
        );
        assert_eq!(Scalar::MAX.to_str_radix(16, 0, false), "7f000000");
    }

    #[test]
    fn test_parse_hexadecimal_upper_case() {
        assert_eq!(Scalar::from_str_radix("0", 16).unwrap(), from_const(0));
        assert_eq!(Scalar::from_str_radix("2A", 16).unwrap(), from_const(42));
        assert_eq!(Scalar::from_str_radix("7F000000", 16).unwrap(), Scalar::MAX);
    }

    #[test]
    fn test_print_hexadecimal_upper_case() {
        assert_eq!(from_const(0).to_str_radix(16, 0, true), "0");
        assert_eq!(from_const(0x13371337).to_str_radix(16, 0, true), "13371337");
        assert_eq!(Scalar::MAX.to_str_radix(16, 0, true), "7F000000");
    }

    #[test]
    fn test_try_to_u8() {
        assert_eq!(from_const(0).try_to_u8().unwrap(), 0);
        assert_eq!(from_const(u8::MAX as u32).try_to_u8().unwrap(), u8::MAX);
        assert!(from_const(u8::MAX as u32 + 1).try_to_u8().is_none());
    }

    #[test]
    fn test_try_to_u16() {
        assert_eq!(from_const(0).try_to_u16().unwrap(), 0);
        assert_eq!(from_const(u16::MAX as u32).try_to_u16().unwrap(), u16::MAX);
        assert!(from_const(u16::MAX as u32 + 1).try_to_u16().is_none());
    }

    #[test]
    fn test_to_u32() {
        assert_eq!(from_const(0).to_u32(), 0);
        assert_eq!(from_const(42).to_u32(), 42);
        assert_eq!(Scalar::MAX.to_u32(), MODULUS - 1);
    }

    #[test]
    fn test_to_le_bytes() {
        assert_eq!(from_const(0x01234567).to_le_bytes(), [103, 69, 35, 1]);
        assert_eq!(Scalar::MAX.to_le_bytes(), [0, 0, 0, 127]);
    }

    #[test]
    fn test_to_be_bytes() {
        assert_eq!(from_const(0x01234567).to_be_bytes(), [1, 35, 69, 103]);
        assert_eq!(Scalar::MAX.to_be_bytes(), [127, 0, 0, 0]);
    }

    #[test]
    fn test_from_u128_mod_n() {
        assert_eq!(Scalar::from_u128_mod_n(0), from_const(0));
        assert_eq!(Scalar::from_u128_mod_n(42), from_const(42));
        assert_eq!(
            Scalar::from_u128_mod_n((1u128 << 127) + 12345),
            from_const(0x211eee7f)
        );
    }

    #[test]
    fn test_from_u256_mod_n() {
        assert_eq!(Scalar::from_u256_mod_n(0.into()), from_const(0));
        assert_eq!(Scalar::from_u256_mod_n(42.into()), from_const(42));
        assert_eq!(Scalar::from_u256_mod_n(MODULUS.into()), from_const(0));
        assert_eq!(
            Scalar::from_u256_mod_n(U256::from(MODULUS) + U256::from(1)),
            from_const(1)
        );
    }

    #[test]
    fn test_to_u64() {
        assert_eq!(from_const(0).to_u64(), 0);
        assert_eq!(from_const(42).to_u64(), 42);
        assert_eq!(Scalar::MAX.to_u64(), (MODULUS - 1) as u64);
    }

    #[test]
    fn test_to_u128() {
        assert_eq!(from_const(0).to_u128(), 0);
        assert_eq!(from_const(42).to_u128(), 42);
        assert_eq!(Scalar::MAX.to_u128(), (MODULUS - 1) as u128);
    }

    #[test]
    fn test_to_u256() {
        assert_eq!(from_const(0).to_u256(), U256::from(0));
        assert_eq!(from_const(42).to_u256(), U256::from(42));
        assert_eq!(Scalar::MAX.to_u256(), U256::from(MODULUS - 1));
    }

    #[test]
    fn test_to_u512() {
        assert_eq!(from_const(0).to_u512(), U512::from(0));
        assert_eq!(from_const(42).to_u512(), U512::from(42));
        assert_eq!(Scalar::MAX.to_u512(), U512::from(MODULUS - 1));
    }

    #[test]
    fn test_from_str() {
        assert_eq!("0".parse::<Scalar>().unwrap(), Scalar::ZERO);
        assert_eq!("1".parse::<Scalar>().unwrap(), Scalar::ONE);
        assert_eq!("42".parse::<Scalar>().unwrap(), from_const(42));
        assert_eq!("0x2a".parse::<Scalar>().unwrap(), from_const(42));
        assert_eq!("0X2A".parse::<Scalar>().unwrap(), from_const(42));
        assert_eq!("0b101010".parse::<Scalar>().unwrap(), from_const(42));
        assert_eq!("0B101010".parse::<Scalar>().unwrap(), from_const(42));
        assert_eq!("0o52".parse::<Scalar>().unwrap(), from_const(42));
        assert_eq!("0O52".parse::<Scalar>().unwrap(), from_const(42));
        assert_eq!("052".parse::<Scalar>().unwrap(), from_const(42));
        assert_eq!("0x7f000000".parse::<Scalar>().unwrap(), Scalar::MAX);
        assert!("0x7f000001".parse::<Scalar>().is_err());
        assert!("2130706433".parse::<Scalar>().is_err());
    }

    #[test]
    fn test_parse_scalar_hex() {
        assert_eq!(parse_scalar("0x2a"), from_const(42));
        assert_eq!(parse_scalar("0x7f000000"), Scalar::MAX);
    }

    #[test]
    fn test_multiplicative_generator() {
        assert_eq!(Scalar::MULTIPLICATIVE_GENERATOR, from_const(3));
        assert_eq!(
            Scalar::MULTIPLICATIVE_GENERATOR.pow(Scalar::MAX / from_const(1u32 << Scalar::S)),
            Scalar::ROOT_OF_UNITY
        );
        assert_ne!(
            Scalar::MULTIPLICATIVE_GENERATOR.pow(from_const(1u32 << Scalar::S)),
            Scalar::ONE
        );
    }

    #[test]
    fn test_minus_two() {
        assert_eq!(Scalar::MINUS_TWO, -from_const(2));
        assert_eq!(
            from_const(42).invert_unwrap(),
            from_const(42).pow(Scalar::MINUS_TWO)
        );
    }

    #[test]
    fn test_two_inv() {
        assert_eq!(Scalar::TWO_INV, from_const(2).invert_unwrap());
        assert_eq!(Scalar::TWO_INV.invert_unwrap(), from_const(2));
    }

    #[test]
    fn test_root_of_unity() {
        for i in 0..Scalar::S {
            assert_ne!(
                Scalar::ROOT_OF_UNITY.pow(from_const(1u32 << i)),
                Scalar::ONE
            );
        }
        assert_eq!(
            Scalar::ROOT_OF_UNITY.pow(from_const(1u32 << Scalar::S)),
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
            Scalar::MULTIPLICATIVE_GENERATOR.pow(from_const(1u32 << Scalar::S))
        );
    }

    #[test]
    fn test_alpha() {
        assert_eq!(Scalar::ALPHA, 3);
        let max: U256 = Scalar::MAX.to_u256();
        let alpha: U256 = Scalar::ALPHA.into();
        assert_ne!(max % alpha, U256::zero());
    }
}
