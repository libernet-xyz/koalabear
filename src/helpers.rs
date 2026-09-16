/// The order of the KoalaBear field, `0x7F000001`.
pub(crate) const MODULUS: u32 = 0x7F000001;

/// Barrett reduction factor, ie. `floor(2^62 / MODULUS)`.
///
/// The shift is 62 because the products we reduce are always lower than `MODULUS^2`, which in turn
/// is lower than `2^62`. Under that bound the estimated quotient is never off by more than 1, so a
/// single conditional subtraction is enough to bring the remainder back into `[0, MODULUS)`.
const BARRETT_FACTOR: u64 = ((1u128 << 62) / MODULUS as u128) as u64;

/// KoalaBear addition.
#[inline]
pub(crate) const fn kb_add(lhs: u32, rhs: u32) -> u32 {
    let sum = lhs + rhs;
    if sum < MODULUS { sum } else { sum - MODULUS }
}

/// KoalaBear subtraction.
#[inline]
pub(crate) const fn kb_sub(lhs: u32, rhs: u32) -> u32 {
    let (difference, borrow) = lhs.overflowing_sub(rhs);
    if borrow {
        difference.wrapping_add(MODULUS)
    } else {
        difference
    }
}

/// KoalaBear multiplication.
///
/// Reduces the 62-bit product with Barrett reduction, which takes two extra multiplications and one
/// conditional subtraction.
#[inline]
pub(crate) const fn kb_mul(lhs: u32, rhs: u32) -> u32 {
    let product = (lhs as u64) * (rhs as u64);
    let quotient = (((product as u128) * BARRETT_FACTOR as u128) >> 62) as u64;
    let remainder = (product - quotient * MODULUS as u64) as u32;
    if remainder < MODULUS {
        remainder
    } else {
        remainder - MODULUS
    }
}
