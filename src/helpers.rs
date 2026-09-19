/// The order of the KoalaBear field, `0x7F000001`.
pub(crate) const MODULUS: u32 = 0x7F000001;

/// `2^32 mod MODULUS`, ie. one in Montgomery form.
const R: u32 = ((1u64 << 32) % MODULUS as u64) as u32;

/// `2^64 mod MODULUS`, ie. `R` in Montgomery form. Multiplying by it converts to Montgomery form.
const R2: u32 = ((R as u64 * R as u64) % MODULUS as u64) as u32;

/// `-MODULUS^-1 mod 2^32`.
const P_INV: u32 = 0x7EFFFFFF;

/// The quadratic non-residue used to build the extension: `X^2 = QUADRATIC_NON_RESIDUE` in the base
/// field.
pub(crate) const QUADRATIC_NON_RESIDUE: u32 = 3;

/// Upper-case characters used in textual representations.
pub(crate) static CHARACTERS_UPPER_CASE: &'static [u8] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";

/// Lower-case characters used in textual representations.
pub(crate) static CHARACTERS_LOWER_CASE: &'static [u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";

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

/// Montgomery reduction via REDC (see
/// https://en.wikipedia.org/wiki/Montgomery_modular_multiplication#The_REDC_algorithm).
///
/// REQUIRES: `value` must be strictly less than `MODULUS * 2^32`.
#[inline]
const fn kb_reduce(value: u64) -> u32 {
    debug_assert!(value < ((MODULUS as u64) << 32));
    let m = (value as u32).wrapping_mul(P_INV) as u64;
    let reduced = ((value + m * MODULUS as u64) >> 32) as u32;
    if reduced < MODULUS {
        reduced
    } else {
        reduced - MODULUS
    }
}

/// Montgomery reduction over a wider domain, `[0, MODULUS << 33)`.
///
/// The algorithm works by bringing the `value` back into the `[0, MODULUS << 32)` range and then
/// invoking [`kb_reduce`].
///
/// Note that this wide range is suitable to hold the sum of at most 4 unreduced products: each
/// product spans 31+31=62 bits, the sum of two spans 63 bits, and the sum of two sums spans 64
/// bits.
#[inline]
const fn kb_reduce_wide(value: u64) -> u32 {
    debug_assert!(value < ((MODULUS as u64) << 33));
    const MODULUS_SHIFTED: u64 = (MODULUS as u64) << 32;
    let value = if value < MODULUS_SHIFTED {
        value
    } else {
        // MODULUS_SHIFTED is a multiple of MODULUS, so subtracting it doesn't change the residue.
        value - MODULUS_SHIFTED
    };
    kb_reduce(value)
}

/// KoalaBear multiplication, ie. `lhs * rhs * R^-1 mod MODULUS`.
#[inline]
pub(crate) const fn kb_mul(lhs: u32, rhs: u32) -> u32 {
    kb_reduce((lhs as u64) * (rhs as u64))
}

/// Converts `value` to Montgomery form, reducing it modulo [`MODULUS`] if needed.
#[inline]
pub(crate) const fn kb_to_montgomery(value: u32) -> u32 {
    kb_mul(value, R2)
}

/// Converts `value` from Montgomery form back to its canonical representative in `[0, MODULUS)`.
#[inline]
pub(crate) const fn kb_from_montgomery(value: u32) -> u32 {
    kb_reduce(value as u64)
}

/// Portable implementations of the extension field operations, used on targets without a SIMD
/// implementation and as the reference the SIMD ones are tested against.
#[allow(dead_code)]
mod scalar {
    use super::{QUADRATIC_NON_RESIDUE, kb_add, kb_mul, kb_reduce, kb_reduce_wide, kb_sub};

    /// Adds two KoalaBear^2 scalars.
    ///
    /// All coefficients are in big-endian order.
    #[inline]
    pub(crate) const fn kb_add2(lhs: [u32; 2], rhs: [u32; 2]) -> [u32; 2] {
        let [a0, a1] = lhs;
        let [b0, b1] = rhs;
        let c0 = kb_add(a0, b0);
        let c1 = kb_add(a1, b1);
        [c0, c1]
    }

    /// Subtracts a KoalaBear^2 scalar from another.
    ///
    /// All coefficients are in big-endian order.
    #[inline]
    pub(crate) const fn kb_sub2(lhs: [u32; 2], rhs: [u32; 2]) -> [u32; 2] {
        let [a0, a1] = lhs;
        let [b0, b1] = rhs;
        let c0 = kb_sub(a0, b0);
        let c1 = kb_sub(a1, b1);
        [c0, c1]
    }

    /// Multiplies two KoalaBear^2 scalars.
    ///
    /// Multiplication is defined over KoalaBear using the tower construction:
    ///
    ///   (lhs[0] * X + lhs[1]) * (rhs[0] * X + rhs[1])
    ///
    /// where `X^2` equals [`QUADRATIC_NON_RESIDUE`].
    ///
    /// All coefficients are in big-endian order.
    #[inline]
    pub(crate) const fn kb_mul2(lhs: [u32; 2], rhs: [u32; 2]) -> [u32; 2] {
        const { assert!(QUADRATIC_NON_RESIDUE == 3) }
        let [a0, a1] = lhs;
        let [b0, b1] = rhs;
        let (a0, a1) = (a0 as u64, a1 as u64);
        let (b0, b1) = (b0 as u64, b1 as u64);
        let c0 = kb_reduce(a0 * b1 + a1 * b0);
        let c1 = kb_reduce_wide(3 * (a0 * b0) + a1 * b1);
        [c0, c1]
    }

    /// Multiplies a KoalaBear^2 scalar by a KoalaBear scalar.
    ///
    /// All coefficients are in big-endian order.
    #[inline]
    pub(crate) const fn kb_mul2x1(lhs: [u32; 2], rhs: u32) -> [u32; 2] {
        let [a0, a1] = lhs;
        let c0 = kb_mul(a0, rhs);
        let c1 = kb_mul(a1, rhs);
        [c0, c1]
    }

    /// Adds two KoalaBear^4 scalars.
    ///
    /// All coefficients are in big-endian order.
    #[inline]
    pub(crate) const fn kb_add4(lhs: [u32; 4], rhs: [u32; 4]) -> [u32; 4] {
        let [a0, a1, a2, a3] = lhs;
        let [b0, b1, b2, b3] = rhs;
        let c0 = kb_add(a0, b0);
        let c1 = kb_add(a1, b1);
        let c2 = kb_add(a2, b2);
        let c3 = kb_add(a3, b3);
        [c0, c1, c2, c3]
    }

    /// Subtracts a KoalaBear^4 scalar from another.
    ///
    /// All coefficients are in big-endian order.
    #[inline]
    pub(crate) const fn kb_sub4(lhs: [u32; 4], rhs: [u32; 4]) -> [u32; 4] {
        let [a0, a1, a2, a3] = lhs;
        let [b0, b1, b2, b3] = rhs;
        let c0 = kb_sub(a0, b0);
        let c1 = kb_sub(a1, b1);
        let c2 = kb_sub(a2, b2);
        let c3 = kb_sub(a3, b3);
        [c0, c1, c2, c3]
    }

    /// Multiplies two KoalaBear^4 scalars.
    ///
    /// Multiplication is defined over KoalaBear^2 using the tower construction:
    ///
    ///   ((lhs[0] * X + lhs[1]) * Y + (lhs[2] * X + lhs[3])) *
    ///     ((rhs[0] * X + rhs[1]) * Y + (rhs[2] * X + rhs[3]))
    ///
    /// where `X^2` and `Y^4` equal [`QUADRATIC_NON_RESIDUE`].
    ///
    /// All coefficients are in big-endian order.
    #[inline]
    pub(crate) const fn kb_mul4(lhs: [u32; 4], rhs: [u32; 4]) -> [u32; 4] {
        const { assert!(QUADRATIC_NON_RESIDUE == 3) }
        let [a0, a1, a2, a3] = lhs;
        let [b0, b1, b2, b3] = rhs;
        let a0_3 = kb_add(kb_add(a0, a0), a0);
        let a1_3 = kb_add(kb_add(a1, a1), a1);
        let a2_3 = kb_add(kb_add(a2, a2), a2);
        let (a0, a1, a2, a3) = (a0 as u64, a1 as u64, a2 as u64, a3 as u64);
        let (b0, b1, b2, b3) = (b0 as u64, b1 as u64, b2 as u64, b3 as u64);
        let (a0_3, a1_3, a2_3) = (a0_3 as u64, a1_3 as u64, a2_3 as u64);
        let c0 = kb_reduce_wide(a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0);
        let c1 = kb_reduce_wide(a1 * b3 + a3 * b1 + a0_3 * b2 + a2_3 * b0);
        let c2 = kb_reduce_wide(a1 * b1 + a2 * b3 + a3 * b2 + a0_3 * b0);
        let c3 = kb_reduce_wide(a3 * b3 + a0_3 * b1 + a1_3 * b0 + a2_3 * b2);
        [c0, c1, c2, c3]
    }

    /// Multiplies a KoalaBear^4 scalar by a KoalaBear scalar.
    ///
    /// All coefficients are in big-endian order.
    #[inline]
    pub(crate) const fn kb_mul4x1(lhs: [u32; 4], rhs: u32) -> [u32; 4] {
        let [a0, a1, a2, a3] = lhs;
        let c0 = kb_mul(a0, rhs);
        let c1 = kb_mul(a1, rhs);
        let c2 = kb_mul(a2, rhs);
        let c3 = kb_mul(a3, rhs);
        [c0, c1, c2, c3]
    }

    /// Adds two KoalaBear^8 scalars.
    ///
    /// All coefficients are in big-endian order.
    #[inline]
    pub(crate) const fn kb_add8(lhs: [u32; 8], rhs: [u32; 8]) -> [u32; 8] {
        let [a0, a1, a2, a3, a4, a5, a6, a7] = lhs;
        let [b0, b1, b2, b3, b4, b5, b6, b7] = rhs;
        let c0 = kb_add(a0, b0);
        let c1 = kb_add(a1, b1);
        let c2 = kb_add(a2, b2);
        let c3 = kb_add(a3, b3);
        let c4 = kb_add(a4, b4);
        let c5 = kb_add(a5, b5);
        let c6 = kb_add(a6, b6);
        let c7 = kb_add(a7, b7);
        [c0, c1, c2, c3, c4, c5, c6, c7]
    }

    /// Subtracts a KoalaBear^8 scalar from another.
    ///
    /// All coefficients are in big-endian order.
    #[inline]
    pub(crate) const fn kb_sub8(lhs: [u32; 8], rhs: [u32; 8]) -> [u32; 8] {
        let [a0, a1, a2, a3, a4, a5, a6, a7] = lhs;
        let [b0, b1, b2, b3, b4, b5, b6, b7] = rhs;
        let c0 = kb_sub(a0, b0);
        let c1 = kb_sub(a1, b1);
        let c2 = kb_sub(a2, b2);
        let c3 = kb_sub(a3, b3);
        let c4 = kb_sub(a4, b4);
        let c5 = kb_sub(a5, b5);
        let c6 = kb_sub(a6, b6);
        let c7 = kb_sub(a7, b7);
        [c0, c1, c2, c3, c4, c5, c6, c7]
    }

    /// Multiplies two KoalaBear^8 scalars.
    ///
    /// Multiplication is defined over KoalaBear^4 using the tower construction.
    ///
    /// All coefficients are in big-endian order.
    #[inline]
    pub(crate) const fn kb_mul8(lhs: [u32; 8], rhs: [u32; 8]) -> [u32; 8] {
        const { assert!(QUADRATIC_NON_RESIDUE == 3) }
        let [a0, a1, a2, a3, a4, a5, a6, a7] = lhs;
        let [b0, b1, b2, b3, b4, b5, b6, b7] = rhs;
        let ac = kb_mul4([a0, a1, a2, a3], [b0, b1, b2, b3]);
        let bd = kb_mul4([a4, a5, a6, a7], [b4, b5, b6, b7]);
        let s = kb_mul4(
            kb_add4([a0, a1, a2, a3], [a4, a5, a6, a7]),
            kb_add4([b0, b1, b2, b3], [b4, b5, b6, b7]),
        );
        let [ac0, ac1, ac2, ac3] = ac;
        let acy = [ac2, ac3, ac1, kb_add(kb_add(ac0, ac0), ac0)];
        let [c0, c1, c2, c3] = kb_sub4(kb_sub4(s, ac), bd);
        let [c4, c5, c6, c7] = kb_add4(bd, acy);
        [c0, c1, c2, c3, c4, c5, c6, c7]
    }

    /// Multiplies a KoalaBear^8 scalar by a KoalaBear scalar.
    ///
    /// All coefficients are in big-endian order.
    #[inline]
    pub(crate) const fn kb_mul8x1(lhs: [u32; 8], rhs: u32) -> [u32; 8] {
        let [a0, a1, a2, a3, a4, a5, a6, a7] = lhs;
        let c0 = kb_mul(a0, rhs);
        let c1 = kb_mul(a1, rhs);
        let c2 = kb_mul(a2, rhs);
        let c3 = kb_mul(a3, rhs);
        let c4 = kb_mul(a4, rhs);
        let c5 = kb_mul(a5, rhs);
        let c6 = kb_mul(a6, rhs);
        let c7 = kb_mul(a7, rhs);
        [c0, c1, c2, c3, c4, c5, c6, c7]
    }
}

/// SSE2 implementations of the extension field operations. SSE2 is part of the x86-64 baseline,
/// so these need no runtime detection.
#[cfg(target_arch = "x86_64")]
mod x86_64 {
    use super::{MODULUS, P_INV, QUADRATIC_NON_RESIDUE, kb_add};
    use core::arch::x86_64::*;

    /// Subtracts `modulus` from every 32-bit lane of `value` that is not lower than it, provided
    /// that the difference fits in a signed 32-bit integer (eg. a value lower than `2 * MODULUS`
    /// against `MODULUS`). Lanes where `modulus` is zero are left untouched.
    ///
    /// SSE2 has no unsigned 32-bit comparison nor minimum, but since the difference fits in a
    /// signed integer its sign bit is exactly the borrow, which is broadcast to the whole lane to
    /// mask `modulus`. This is written as a subtraction from `value` rather than an addition to the
    /// difference so that LLVM doesn't rewrite it into a select, which costs an instruction more.
    #[inline]
    fn subtract_modulus(value: __m128i, modulus: __m128i) -> __m128i {
        // SAFETY: SSE2 is part of the x86-64 baseline, so the intrinsics are always available.
        unsafe {
            let borrow = _mm_srai_epi32::<31>(_mm_sub_epi32(value, modulus));
            _mm_sub_epi32(value, _mm_andnot_si128(borrow, modulus))
        }
    }

    /// Adds `lhs` and `rhs` lane by lane: each sum is lower than `2 * MODULUS`, so a single
    /// [`subtract_modulus`] reduces it.
    #[inline]
    fn add(lhs: __m128i, rhs: __m128i) -> __m128i {
        // SAFETY: SSE2 is part of the x86-64 baseline, so the intrinsics are always available.
        unsafe { subtract_modulus(_mm_add_epi32(lhs, rhs), _mm_set1_epi32(MODULUS as i32)) }
    }

    /// Subtracts `rhs` from `lhs` lane by lane. A difference that borrows wraps around to at least
    /// `2^32 - MODULUS`, which is beyond `2^31` since `MODULUS < 2^31`, while one that does not is
    /// lower than `MODULUS < 2^31`: the borrow is exactly the sign bit of the difference, and
    /// `MODULUS` is added back where it is set.
    #[inline]
    fn sub(lhs: __m128i, rhs: __m128i) -> __m128i {
        // SAFETY: SSE2 is part of the x86-64 baseline, so the intrinsics are always available.
        unsafe {
            let difference = _mm_sub_epi32(lhs, rhs);
            let borrow = _mm_srai_epi32::<31>(difference);
            _mm_add_epi32(
                difference,
                _mm_and_si128(_mm_set1_epi32(MODULUS as i32), borrow),
            )
        }
    }

    /// Multiplies every lane, lower than [`MODULUS`], by [`QUADRATIC_NON_RESIDUE`].
    ///
    /// The triple wraps around `2^32` but the reduced triple doesn't, so the wrapped triple is
    /// corrected by subtracting `MODULUS` once where the value is at least `MODULUS / 3` and once
    /// more where it is at least `2 * MODULUS / 3`, with signed comparisons since every value
    /// involved is lower than `2^31`. This costs two instructions less than two modular additions.
    #[inline]
    fn triple(value: __m128i) -> __m128i {
        const { assert!(QUADRATIC_NON_RESIDUE == 3) }
        const ONCE_THRESHOLD: i32 = MODULUS.div_ceil(3) as i32 - 1;
        const TWICE_THRESHOLD: i32 = (2 * MODULUS).div_ceil(3) as i32 - 1;
        // SAFETY: SSE2 is part of the x86-64 baseline, so the intrinsics are always available.
        unsafe {
            let modulus = _mm_set1_epi32(MODULUS as i32);
            let tripled = _mm_add_epi32(_mm_add_epi32(value, value), value);
            let once = _mm_cmpgt_epi32(value, _mm_set1_epi32(ONCE_THRESHOLD));
            let twice = _mm_cmpgt_epi32(value, _mm_set1_epi32(TWICE_THRESHOLD));
            _mm_sub_epi32(
                _mm_sub_epi32(tripled, _mm_and_si128(modulus, once)),
                _mm_and_si128(modulus, twice),
            )
        }
    }

    /// Montgomery-reduces both 64-bit lanes of `sum`, each of which must be lower than
    /// `MODULUS * 2^32`, leaving results lower than `2 * MODULUS` (rather than `MODULUS`) in the
    /// low 32-bit halves of the lanes (ie. 32-bit lanes 0 and 2) with zeroed high halves.
    ///
    /// `_mm_mul_epu32` only reads the low 32 bits of each 64-bit lane, which is exactly what both
    /// steps of the reduction need: the low word of `sum` to derive the factor, then the factor
    /// itself. The final conditional subtraction is left to the caller so that the results of
    /// several reductions can be gathered into one register and corrected all at once.
    #[inline]
    fn reduce_partially(sum: __m128i) -> __m128i {
        // SAFETY: SSE2 is part of the x86-64 baseline, so the intrinsics are always available.
        unsafe {
            let factor = _mm_mul_epu32(sum, _mm_set1_epi32(P_INV as i32));
            let product = _mm_mul_epu32(factor, _mm_set1_epi32(MODULUS as i32));
            _mm_srli_epi64::<32>(_mm_add_epi64(sum, product))
        }
    }

    /// Montgomery-reduces both 64-bit lanes of `sum`, each of which must be lower than
    /// `MODULUS * 2^32`, leaving the results in the low 32-bit halves of the lanes (ie. 32-bit
    /// lanes 0 and 2) with zeroed high halves.
    ///
    /// This is [`reduce_partially`] followed by the single [`subtract_modulus`] its results need;
    /// on the zeroed high halves it is a no-op.
    #[inline]
    fn reduce(sum: __m128i) -> __m128i {
        // SAFETY: SSE2 is part of the x86-64 baseline, so the intrinsics are always available.
        unsafe { subtract_modulus(reduce_partially(sum), _mm_set1_epi32(MODULUS as i32)) }
    }

    /// Montgomery-reduces both 64-bit lanes of `sum`, each of which must be lower than
    /// `MODULUS * 2^33`, leaving the results in the low 32-bit halves of the lanes (ie. 32-bit
    /// lanes 0 and 2) with zeroed high halves.
    ///
    /// This is [`super::kb_reduce_wide`] lane by lane: `MODULUS` is subtracted from every high word
    /// that is not lower than it, ie. `MODULUS * 2^32` from the 64-bit lane, which doesn't change
    /// its residue and brings it into the domain of [`reduce`].
    #[inline]
    fn reduce_wide(sum: __m128i) -> __m128i {
        // SAFETY: SSE2 is part of the x86-64 baseline, so the intrinsics are always available.
        unsafe {
            let modulus_high = _mm_set_epi32(MODULUS as i32, 0, MODULUS as i32, 0);
            reduce(subtract_modulus(sum, modulus_high))
        }
    }

    /// SSE2 version of [`super::scalar::kb_add2`]: the two coefficients are added in 32-bit lanes 0
    /// and 1 by [`add`].
    #[inline]
    pub(crate) fn kb_add2(lhs: [u32; 2], rhs: [u32; 2]) -> [u32; 2] {
        // SAFETY: SSE2 is part of the x86-64 baseline, so the intrinsics are always available, and
        // `_mm_loadl_epi64` reads 8 bytes, the size of the arrays, with no alignment requirement.
        unsafe {
            let reduced = add(
                _mm_loadl_epi64(lhs.as_ptr().cast()),
                _mm_loadl_epi64(rhs.as_ptr().cast()),
            );
            let packed = _mm_cvtsi128_si64(reduced) as u64;
            let c0 = packed as u32;
            let c1 = (packed >> 32) as u32;
            [c0, c1]
        }
    }

    /// SSE2 version of [`super::scalar::kb_sub2`]: the two coefficients are subtracted in 32-bit
    /// lanes 0 and 1 by [`sub`].
    #[inline]
    pub(crate) fn kb_sub2(lhs: [u32; 2], rhs: [u32; 2]) -> [u32; 2] {
        // SAFETY: SSE2 is part of the x86-64 baseline, so the intrinsics are always available, and
        // `_mm_loadl_epi64` reads 8 bytes, the size of the arrays, with no alignment requirement.
        unsafe {
            let reduced = sub(
                _mm_loadl_epi64(lhs.as_ptr().cast()),
                _mm_loadl_epi64(rhs.as_ptr().cast()),
            );
            let packed = _mm_cvtsi128_si64(reduced) as u64;
            let c0 = packed as u32;
            let c1 = (packed >> 32) as u32;
            [c0, c1]
        }
    }

    /// SSE2 version of [`super::scalar::kb_mul2`].
    ///
    /// `_mm_mul_epu32` multiplies the 32-bit lanes 0 and 2 of its operands into two 64-bit lanes,
    /// so the four products are computed two at a time with the operands laid out such that the two
    /// products of each coefficient land in the same lane. The product to scale by
    /// `QUADRATIC_NON_RESIDUE` is alone in its lane of the first pair, where it's scaled with a
    /// shift and a masked addition; one 64-bit addition then yields both sums, and both are reduced
    /// at once.
    #[inline]
    pub(crate) fn kb_mul2(lhs: [u32; 2], rhs: [u32; 2]) -> [u32; 2] {
        const { assert!(QUADRATIC_NON_RESIDUE == 3) }
        // SAFETY: SSE2 is part of the x86-64 baseline, so the intrinsics are always available, and
        // `_mm_loadl_epi64` reads 8 bytes, the size of the arrays, with no alignment requirement.
        unsafe {
            let lhs = _mm_loadl_epi64(lhs.as_ptr().cast());
            let rhs = _mm_loadl_epi64(rhs.as_ptr().cast());
            // Lanes 0 and 2 (the immediates are given most significant lane first).
            let first = _mm_mul_epu32(
                _mm_shuffle_epi32::<0b00_00_00_00>(lhs),
                _mm_shuffle_epi32::<0b00_00_01_01>(rhs),
            ); // [a0 * b1, a0 * b0]
            let second = _mm_mul_epu32(
                _mm_shuffle_epi32::<0b01_01_01_01>(lhs),
                _mm_shuffle_epi32::<0b01_01_00_00>(rhs),
            ); // [a1 * b0, a1 * b1]
            let scaled = _mm_add_epi64(
                first,
                _mm_and_si128(_mm_slli_epi64::<1>(first), _mm_set_epi64x(-1, 0)),
            ); // [a0 * b1, 3 * a0 * b0]
            let reduced = reduce_wide(_mm_add_epi64(scaled, second));
            let packed = _mm_cvtsi128_si64(_mm_shuffle_epi32::<0b00_00_10_00>(reduced)) as u64;
            let c0 = packed as u32;
            let c1 = (packed >> 32) as u32;
            [c0, c1]
        }
    }

    /// SSE2 version of [`super::scalar::kb_mul2x1`]: the two coefficients are spread to lanes 0
    /// and 2, multiplied by `rhs` in a single `_mm_mul_epu32`, and reduced at once.
    #[inline]
    pub(crate) fn kb_mul2x1(lhs: [u32; 2], rhs: u32) -> [u32; 2] {
        // SAFETY: SSE2 is part of the x86-64 baseline, so the intrinsics are always available, and
        // `_mm_loadl_epi64` reads 8 bytes, the size of the array, with no alignment requirement.
        unsafe {
            // (a0, ., a1, .) (the immediate is given most significant lane first).
            let lhs = _mm_shuffle_epi32::<0b01_01_00_00>(_mm_loadl_epi64(lhs.as_ptr().cast()));
            let reduced = reduce(_mm_mul_epu32(lhs, _mm_set1_epi32(rhs as i32)));
            let packed = _mm_cvtsi128_si64(_mm_shuffle_epi32::<0b00_00_10_00>(reduced)) as u64;
            let c0 = packed as u32;
            let c1 = (packed >> 32) as u32;
            [c0, c1]
        }
    }

    /// SSE2 version of [`super::scalar::kb_add4`]: the four coefficients occupy a full 128-bit
    /// register, one per lane, so this is a single [`add`].
    #[inline]
    pub(crate) fn kb_add4(lhs: [u32; 4], rhs: [u32; 4]) -> [u32; 4] {
        // SAFETY: SSE2 is part of the x86-64 baseline, so the intrinsics are always available, and
        // the arrays are 16 bytes, the size of a `__m128i`, while the unaligned load and store have
        // no alignment requirement.
        unsafe {
            let reduced = add(
                _mm_loadu_si128(lhs.as_ptr().cast()),
                _mm_loadu_si128(rhs.as_ptr().cast()),
            );
            let mut out = [0u32; 4];
            _mm_storeu_si128(out.as_mut_ptr().cast(), reduced);
            out
        }
    }

    /// SSE2 version of [`super::scalar::kb_sub4`]: the four coefficients occupy a full 128-bit
    /// register, one per lane, so this is a single [`sub`].
    #[inline]
    pub(crate) fn kb_sub4(lhs: [u32; 4], rhs: [u32; 4]) -> [u32; 4] {
        // SAFETY: SSE2 is part of the x86-64 baseline, so the intrinsics are always available, and
        // the arrays are 16 bytes, the size of a `__m128i`, while the unaligned load and store
        // have no alignment requirement.
        unsafe {
            let reduced = sub(
                _mm_loadu_si128(lhs.as_ptr().cast()),
                _mm_loadu_si128(rhs.as_ptr().cast()),
            );
            let mut out = [0u32; 4];
            _mm_storeu_si128(out.as_mut_ptr().cast(), reduced);
            out
        }
    }

    /// SSE2 version of [`super::scalar::kb_mul4`].
    ///
    /// `_mm_mul_epu32` multiplies the 32-bit lanes 0 and 2 of its operands into two 64-bit lanes
    /// and ignores lanes 1 and 3, so the 16 products are computed by eight multiplications whose
    /// operands pair one product of `c0` with one of `c1`, or one of `c2` with one of `c3`, in
    /// lanes 0 and 2: the products accumulate into a `(c0, c1)` and a `(c2, c3)` register that are
    /// then reduced two coefficients at a time.
    ///
    /// The products are paired such that `rhs` (whose lanes 0 and 2 are `b0` and `b2`) and
    /// `tripled` (`3 * a0` and `3 * a2`) are used as they are, and every other operand is a single
    /// shuffle of `plain`, `tripled` or `rhs`: the two-source ones are `shufps`, the only one SSE2
    /// offers (hence the casts), which takes its low lanes from one source and its high lanes from
    /// the other.
    #[inline]
    pub(crate) fn kb_mul4(lhs: [u32; 4], rhs: [u32; 4]) -> [u32; 4] {
        // SAFETY: SSE2 is part of the x86-64 baseline, so the intrinsics are always available, and
        // the arrays are 16 bytes, the size of a `__m128i`, while the unaligned load and store have
        // no alignment requirement.
        unsafe {
            let plain = _mm_loadu_si128(lhs.as_ptr().cast());
            let rhs = _mm_loadu_si128(rhs.as_ptr().cast());
            let tripled = triple(plain);
            let (plain_ps, tripled_ps) = (_mm_castsi128_ps(plain), _mm_castsi128_ps(tripled));
            // The operands, as their lanes 0 and 2 (the immediates are given most significant lane
            // first).
            let lhs_a1_a3 = _mm_srli_epi64::<32>(plain);
            let lhs_a2_a1 = _mm_shuffle_epi32::<0b01_01_10_10>(plain);
            let lhs_a3_3a0 =
                _mm_castps_si128(_mm_shuffle_ps::<0b00_00_11_11>(plain_ps, tripled_ps));
            let lhs_a0_3a2 =
                _mm_castps_si128(_mm_shuffle_ps::<0b10_10_00_00>(plain_ps, tripled_ps));
            let lhs_a2_3a1 =
                _mm_castps_si128(_mm_shuffle_ps::<0b01_01_10_10>(plain_ps, tripled_ps));
            let rhs_b1_b3 = _mm_srli_epi64::<32>(rhs);
            let rhs_b3_b0 = _mm_shuffle_epi32::<0b00_00_11_11>(rhs);
            let rhs_b2_b1 = _mm_shuffle_epi32::<0b01_01_10_10>(rhs);
            // [a2 * b1 + a3 * b0 + a0 * b3 + a1 * b2,
            //  a1 * b3 + 3 * a0 * b2 + 3 * a2 * b0 + a3 * b1]
            let sum01 = _mm_add_epi64(
                _mm_add_epi64(
                    _mm_mul_epu32(lhs_a2_a1, rhs_b1_b3),
                    _mm_mul_epu32(lhs_a3_3a0, rhs),
                ),
                _mm_add_epi64(
                    _mm_mul_epu32(lhs_a0_3a2, rhs_b3_b0),
                    _mm_mul_epu32(lhs_a1_a3, rhs_b2_b1),
                ),
            );
            // [a1 * b1 + 3 * a0 * b0 + a2 * b3 + a3 * b2,
            //  a3 * b3 + 3 * a2 * b2 + 3 * a1 * b0 + 3 * a0 * b1]
            let sum23 = _mm_add_epi64(
                _mm_add_epi64(
                    _mm_mul_epu32(lhs_a1_a3, rhs_b1_b3),
                    _mm_mul_epu32(tripled, rhs),
                ),
                _mm_add_epi64(
                    _mm_mul_epu32(lhs_a2_3a1, rhs_b3_b0),
                    _mm_mul_epu32(lhs_a3_3a0, rhs_b2_b1),
                ),
            );
            let reduced01 = reduce_wide(sum01);
            let reduced23 = reduce_wide(sum23);
            // Interleaves `(c0, c1)` and `(c2, c3)`, in lanes 0 and 2 of each, into
            // `(c0, c1, c2, c3)`.
            let reduced = _mm_castps_si128(_mm_shuffle_ps::<0b10_00_10_00>(
                _mm_castsi128_ps(reduced01),
                _mm_castsi128_ps(reduced23),
            ));
            let mut out = [0u32; 4];
            _mm_storeu_si128(out.as_mut_ptr().cast(), reduced);
            out
        }
    }

    /// SSE2 version of [`super::scalar::kb_mul4x1`].
    ///
    /// `_mm_mul_epu32` multiplies lanes 0 and 2, so the even coefficients are multiplied by `rhs`
    /// in place and the odd ones after a shift down; the two pairs of products are reduced with
    /// [`reduce_partially`], gathered back into `(c0, c1, c2, c3)` (the odd ones shifted up into
    /// the zeroed high halves of the even ones), and corrected with a single [`subtract_modulus`].
    #[inline]
    pub(crate) fn kb_mul4x1(lhs: [u32; 4], rhs: u32) -> [u32; 4] {
        // SAFETY: SSE2 is part of the x86-64 baseline, so the intrinsics are always available, and
        // the arrays are 16 bytes, the size of a `__m128i`, while the unaligned load and store have
        // no alignment requirement.
        unsafe {
            let lhs = _mm_loadu_si128(lhs.as_ptr().cast());
            let rhs = _mm_set1_epi32(rhs as i32);
            let even = reduce_partially(_mm_mul_epu32(lhs, rhs));
            let odd = reduce_partially(_mm_mul_epu32(_mm_srli_epi64::<32>(lhs), rhs));
            let gathered = _mm_or_si128(even, _mm_slli_epi64::<32>(odd));
            let reduced = subtract_modulus(gathered, _mm_set1_epi32(MODULUS as i32));
            let mut out = [0u32; 4];
            _mm_storeu_si128(out.as_mut_ptr().cast(), reduced);
            out
        }
    }

    /// SSE2 version of [`super::scalar::kb_add8`].
    #[inline]
    pub(crate) fn kb_add8(lhs: [u32; 8], rhs: [u32; 8]) -> [u32; 8] {
        let [a0, a1, a2, a3, a4, a5, a6, a7] = lhs;
        let [b0, b1, b2, b3, b4, b5, b6, b7] = rhs;
        let [c0, c1, c2, c3] = kb_add4([a0, a1, a2, a3], [b0, b1, b2, b3]);
        let [c4, c5, c6, c7] = kb_add4([a4, a5, a6, a7], [b4, b5, b6, b7]);
        [c0, c1, c2, c3, c4, c5, c6, c7]
    }

    /// SSE2 version of [`super::scalar::kb_sub8`].
    #[inline]
    pub(crate) fn kb_sub8(lhs: [u32; 8], rhs: [u32; 8]) -> [u32; 8] {
        let [a0, a1, a2, a3, a4, a5, a6, a7] = lhs;
        let [b0, b1, b2, b3, b4, b5, b6, b7] = rhs;
        let [c0, c1, c2, c3] = kb_sub4([a0, a1, a2, a3], [b0, b1, b2, b3]);
        let [c4, c5, c6, c7] = kb_sub4([a4, a5, a6, a7], [b4, b5, b6, b7]);
        [c0, c1, c2, c3, c4, c5, c6, c7]
    }

    /// SSE2 version of [`super::scalar::kb_mul8`].
    ///
    /// The same Karatsuba over [`kb_mul4`], [`kb_add4`] and [`kb_sub4`], so the three KoalaBear^4
    /// multiplications and the combining steps all run the SSE2 code above.
    #[inline]
    pub(crate) fn kb_mul8(lhs: [u32; 8], rhs: [u32; 8]) -> [u32; 8] {
        const { assert!(QUADRATIC_NON_RESIDUE == 3) }
        let [a0, a1, a2, a3, a4, a5, a6, a7] = lhs;
        let [b0, b1, b2, b3, b4, b5, b6, b7] = rhs;
        let ac = kb_mul4([a0, a1, a2, a3], [b0, b1, b2, b3]);
        let bd = kb_mul4([a4, a5, a6, a7], [b4, b5, b6, b7]);
        let s = kb_mul4(
            kb_add4([a0, a1, a2, a3], [a4, a5, a6, a7]),
            kb_add4([b0, b1, b2, b3], [b4, b5, b6, b7]),
        );
        let [ac0, ac1, ac2, ac3] = ac;
        let acy = [ac2, ac3, ac1, kb_add(kb_add(ac0, ac0), ac0)];
        let [c0, c1, c2, c3] = kb_sub4(kb_sub4(s, ac), bd);
        let [c4, c5, c6, c7] = kb_add4(bd, acy);
        [c0, c1, c2, c3, c4, c5, c6, c7]
    }

    /// SSE2 version of [`super::scalar::kb_mul8x1`].
    ///
    /// Eight coefficients are two full 128-bit registers, so this is [`kb_mul4x1`] on each half.
    #[inline]
    pub(crate) fn kb_mul8x1(lhs: [u32; 8], rhs: u32) -> [u32; 8] {
        let [a0, a1, a2, a3, a4, a5, a6, a7] = lhs;
        let [c0, c1, c2, c3] = kb_mul4x1([a0, a1, a2, a3], rhs);
        let [c4, c5, c6, c7] = kb_mul4x1([a4, a5, a6, a7], rhs);
        [c0, c1, c2, c3, c4, c5, c6, c7]
    }
}

/// WebAssembly SIMD implementations of the extension field operations. Unlike SSE2 on x86-64,
/// `simd128` is not part of the baseline and must be enabled at compile time with
/// `-C target-feature=+simd128`, in which case these replace the scalar versions.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm32 {
    use super::{MODULUS, P_INV, QUADRATIC_NON_RESIDUE, kb_add};
    use core::arch::wasm32::*;

    /// A `v128` worth of `u32`s, aligned for [`core::ptr::read_volatile`] as a `v128`.
    #[repr(align(16))]
    struct Lanes([u32; 4]);

    /// `MODULUS` in every lane, see [`modulus_splat`].
    static MODULUS_SPLAT: Lanes = Lanes([MODULUS; 4]);

    /// `MODULUS` in every 32-bit lane, for the widening multiplications of the reductions.
    ///
    /// Rust's `extmul` intrinsics are not opaque to LLVM: they are a widening of each operand
    /// followed by a 64-bit multiplication, and LLVM only recovers an `i64x2.extmul_low_i32x4_u`
    /// when both operands are still widenings of vectors when the code is generated. Given a
    /// constant splat it folds the widening into a constant of 64-bit lanes, leaving an `i64x2.mul`
    /// that engines expand into three 32x32->64 multiplications on x86-64. The volatile load keeps
    /// the modulus a vector value; it costs a single load from a fixed address.
    #[inline]
    fn modulus_splat() -> v128 {
        // SAFETY: `MODULUS_SPLAT` is 16 bytes and 16-byte aligned, ie. exactly a `v128`.
        unsafe { core::ptr::read_volatile(MODULUS_SPLAT.0.as_ptr().cast::<v128>()) }
    }

    /// Adds `lhs` and `rhs` lane by lane: the conditional subtraction of `MODULUS` is an unsigned
    /// minimum, since when the subtraction should not happen it wraps around to a value that loses
    /// the comparison.
    #[inline]
    fn add(lhs: v128, rhs: v128) -> v128 {
        let sum = u32x4_add(lhs, rhs);
        u32x4_min(sum, u32x4_sub(sum, u32x4_splat(MODULUS)))
    }

    /// Subtracts `rhs` from `lhs` lane by lane: the conditional addition of `MODULUS` is an
    /// unsigned minimum, since a difference that borrows wraps around to a large value, and adding
    /// `MODULUS` wraps it back to the correct, smaller one.
    #[inline]
    fn sub(lhs: v128, rhs: v128) -> v128 {
        let difference = u32x4_sub(lhs, rhs);
        u32x4_min(difference, u32x4_add(difference, u32x4_splat(MODULUS)))
    }

    /// Multiplies every lane by `QUADRATIC_NON_RESIDUE`, ie. two additions.
    #[inline]
    fn triple(value: v128) -> v128 {
        const { assert!(QUADRATIC_NON_RESIDUE == 3) }
        add(add(value, value), value)
    }

    /// WebAssembly SIMD version of [`super::scalar::kb_add2`]: the two coefficients are added in
    /// 32-bit lanes 0 and 1 by [`add`].
    #[inline]
    pub(crate) fn kb_add2(lhs: [u32; 2], rhs: [u32; 2]) -> [u32; 2] {
        let [a0, a1] = lhs;
        let [b0, b1] = rhs;
        let reduced = add(u32x4(a0, a1, 0, 0), u32x4(b0, b1, 0, 0));
        let c0 = u32x4_extract_lane::<0>(reduced);
        let c1 = u32x4_extract_lane::<1>(reduced);
        [c0, c1]
    }

    /// WebAssembly SIMD version of [`super::scalar::kb_sub2`]: the two coefficients are subtracted
    /// in 32-bit lanes 0 and 1 by [`sub`].
    #[inline]
    pub(crate) fn kb_sub2(lhs: [u32; 2], rhs: [u32; 2]) -> [u32; 2] {
        let [a0, a1] = lhs;
        let [b0, b1] = rhs;
        let reduced = sub(u32x4(a0, a1, 0, 0), u32x4(b0, b1, 0, 0));
        let c0 = u32x4_extract_lane::<0>(reduced);
        let c1 = u32x4_extract_lane::<1>(reduced);
        [c0, c1]
    }

    /// Montgomery-reduces both 64-bit lanes of `sum`, each of which must be lower than
    /// `MODULUS * 2^32`, leaving results lower than `2 * MODULUS` (rather than `MODULUS`) in the
    /// low 32-bit halves of the lanes (ie. 32-bit lanes 0 and 2) with zeroed high halves.
    ///
    /// The low words of the 64-bit lanes are the 32-bit lanes 0 and 2, so the factors computed by
    /// the 32-bit multiplication are gathered from there before the widening multiplication by
    /// `MODULUS`. The final conditional subtraction is left to the caller so that the results of
    /// several reductions can be gathered into one vector and corrected all at once.
    #[inline]
    fn reduce_partially(sum: v128) -> v128 {
        let factor = u32x4_mul(sum, u32x4_splat(P_INV));
        let factor = u32x4_shuffle::<0, 2, 0, 2>(factor, factor);
        let product = u64x2_extmul_low_u32x4(factor, modulus_splat());
        u64x2_shr(u64x2_add(sum, product), 32)
    }

    /// Montgomery-reduces both 64-bit lanes of `sum`, each of which must be lower than
    /// `MODULUS * 2^32`, leaving the results in the low 32-bit halves of the lanes (ie. 32-bit
    /// lanes 0 and 2) with zeroed high halves.
    ///
    /// This is [`reduce_partially`] followed by the single [`add`]-style conditional subtraction
    /// its results need; on the zeroed high halves it is a no-op.
    #[inline]
    fn reduce(sum: v128) -> v128 {
        let reduced = reduce_partially(sum);
        u32x4_min(reduced, u32x4_sub(reduced, u32x4_splat(MODULUS)))
    }

    /// WebAssembly SIMD version of [`super::scalar::kb_mul2`].
    ///
    /// The widening multiplications produce the products of lanes 0 and 1 and those of lanes 2 and
    /// 3 as two pairs of 64-bit lanes, so the operands are laid out such that the two products of
    /// each coefficient land in the same lane: one 64-bit addition then yields both sums, and both
    /// are reduced at once.
    ///
    /// The operands are built with lane shuffles, and the scaled coefficient with vector
    /// additions, for the same reason [`modulus_splat`] exists: LLVM widens each side of a
    /// multiplication in whatever way is cheapest for it, and only recovers an `extmul` when both
    /// sides come out the same way. Operands assembled from scalars, and pairs of lanes that
    /// happen to be the high half of some vector (widened with `extend_high` rather than
    /// `extend_low`), both break that, so the sources are laid out such that no pair used by a
    /// multiplication is the high half of a vector. Scaling the product instead of the coefficient,
    /// as the scalar and SSE2 versions do, would make one operand of every multiplication a pair
    /// of equal lanes, which LLVM turns into a splat and thereby breaks the same way.
    #[inline]
    pub(crate) fn kb_mul2(lhs: [u32; 2], rhs: [u32; 2]) -> [u32; 2] {
        let [a0, a1] = lhs;
        let [b0, b1] = rhs;
        let plain = u32x4(a0, a1, 0, 0);
        let tripled = triple(plain);
        let rhs = u32x4(b0, b1, 0, 0);
        // (a0, 3 * a0, a1, a1) * (b1, b0, b0, b1)
        let lhs = u32x4_shuffle::<0, 4, 1, 1>(plain, tripled);
        let rhs = u32x4_shuffle::<1, 0, 0, 1>(rhs, rhs);
        // [a0 * b1 + a1 * b0, 3 * a0 * b0 + a1 * b1]
        let sum = u64x2_add(
            u64x2_extmul_low_u32x4(lhs, rhs),
            u64x2_extmul_high_u32x4(lhs, rhs),
        );
        let reduced = reduce(sum);
        let c0 = u32x4_extract_lane::<0>(reduced);
        let c1 = u32x4_extract_lane::<2>(reduced);
        [c0, c1]
    }

    /// WebAssembly SIMD version of [`super::scalar::kb_mul2x1`]: the two coefficients are
    /// multiplied by `rhs` in a single widening multiplication and reduced at once.
    #[inline]
    pub(crate) fn kb_mul2x1(lhs: [u32; 2], rhs: u32) -> [u32; 2] {
        let [a0, a1] = lhs;
        let products = u64x2_extmul_low_u32x4(u32x4(a0, a1, 0, 0), u32x4_splat(rhs));
        let reduced = reduce(products);
        let c0 = u32x4_extract_lane::<0>(reduced);
        let c1 = u32x4_extract_lane::<2>(reduced);
        [c0, c1]
    }

    /// WebAssembly SIMD version of [`super::scalar::kb_add4`]: the four coefficients occupy a full
    /// `v128`, one per lane, so this is a single [`add`].
    #[inline]
    pub(crate) fn kb_add4(lhs: [u32; 4], rhs: [u32; 4]) -> [u32; 4] {
        let [a0, a1, a2, a3] = lhs;
        let [b0, b1, b2, b3] = rhs;
        let reduced = add(u32x4(a0, a1, a2, a3), u32x4(b0, b1, b2, b3));
        let mut out = [0u32; 4];
        // SAFETY: `out` is 16 bytes, the same size as a `v128`, and WebAssembly's `v128.store`
        // has no alignment requirement.
        unsafe { v128_store(out.as_mut_ptr().cast(), reduced) };
        out
    }

    /// WebAssembly SIMD version of [`super::scalar::kb_sub4`]: the four coefficients occupy a full
    /// `v128`, one per lane, so this is a single [`sub`].
    #[inline]
    pub(crate) fn kb_sub4(lhs: [u32; 4], rhs: [u32; 4]) -> [u32; 4] {
        let [a0, a1, a2, a3] = lhs;
        let [b0, b1, b2, b3] = rhs;
        let reduced = sub(u32x4(a0, a1, a2, a3), u32x4(b0, b1, b2, b3));
        let mut out = [0u32; 4];
        // SAFETY: `out` is 16 bytes, the same size as a `v128`, and WebAssembly's `v128.store`
        // has no alignment requirement.
        unsafe { v128_store(out.as_mut_ptr().cast(), reduced) };
        out
    }

    /// Montgomery-reduces both 64-bit lanes of `sum`, each of which must be lower than
    /// `MODULUS * 2^33`, leaving the results in the low 32-bit halves of the lanes (ie. 32-bit
    /// lanes 0 and 2) with zeroed high halves.
    ///
    /// This is [`super::kb_reduce_wide`] lane by lane: `MODULUS` is subtracted from every high
    /// word that is not lower than it (the usual unsigned minimum, with zeros in the low words so
    /// that they are left untouched), ie. `MODULUS * 2^32` from the 64-bit lane, which doesn't
    /// change its residue and brings it into the domain of [`reduce`].
    #[inline]
    fn reduce_wide(sum: v128) -> v128 {
        let modulus_high = u32x4(0, MODULUS, 0, MODULUS);
        reduce(u32x4_min(sum, u32x4_sub(sum, modulus_high)))
    }

    /// WebAssembly SIMD version of [`super::scalar::kb_mul4`].
    ///
    /// The 16 products are computed in four rounds of two widening multiplications, with the
    /// operands of each round laid out as `(c0, c2, c1, c3)`: the low multiplication yields one
    /// product of `c0` and one of `c2`, the high one one of `c1` and one of `c3`, so the four
    /// rounds accumulate into a `(c0, c2)` and a `(c1, c3)` vector that are then reduced two
    /// coefficients at a time. Unlike `_mm_mul_epu32`, the widening multiplications use all four
    /// lanes of their operands, so every operand is one shuffle for two products and there is
    /// nothing to gain from the pairing of the SSE2 version.
    ///
    /// The left operands are shuffles of `plain` and `tripled` (its lanes times
    /// `QUADRATIC_NON_RESIDUE`, computed with vector additions), the right ones of `rhs`. The
    /// rounds pair the products such that no pair of lanes fed to a multiplication is the high
    /// half of `plain`, `tripled` or `rhs`; see [`kb_mul2`] for why that and the vector
    /// additions matter. LLVM still merges the shuffles of two of the rounds in a way that
    /// loses two of the eight `extmul`s to `i64x2.mul`, which no pairing avoids.
    #[inline]
    pub(crate) fn kb_mul4(lhs: [u32; 4], rhs: [u32; 4]) -> [u32; 4] {
        let [a0, a1, a2, a3] = lhs;
        let [b0, b1, b2, b3] = rhs;
        let plain = u32x4(a0, a1, a2, a3);
        let tripled = triple(plain);
        let rhs = u32x4(b0, b1, b2, b3);
        // (a0, a1, a1, a3) * (b3, b1, b3, b3)
        let lhs1 = u32x4_shuffle::<0, 1, 1, 3>(plain, tripled);
        let rhs1 = u32x4_shuffle::<3, 1, 3, 3>(rhs, rhs);
        // (a1, a3, a3, 3 * a0) * (b2, b2, b1, b1)
        let lhs2 = u32x4_shuffle::<1, 3, 3, 4>(plain, tripled);
        let rhs2 = u32x4_shuffle::<2, 2, 1, 1>(rhs, rhs);
        // (a2, a2, 3 * a0, 3 * a1) * (b1, b3, b2, b0)
        let lhs3 = u32x4_shuffle::<2, 2, 4, 5>(plain, tripled);
        let rhs3 = u32x4_shuffle::<1, 3, 2, 0>(rhs, rhs);
        // (a3, 3 * a0, 3 * a2, 3 * a2) * (b0, b0, b0, b2)
        let lhs4 = u32x4_shuffle::<3, 4, 6, 6>(plain, tripled);
        let rhs4 = u32x4_shuffle::<0, 0, 0, 2>(rhs, rhs);
        let sum02 = u64x2_add(
            u64x2_add(
                u64x2_extmul_low_u32x4(lhs1, rhs1),
                u64x2_extmul_low_u32x4(lhs2, rhs2),
            ),
            u64x2_add(
                u64x2_extmul_low_u32x4(lhs3, rhs3),
                u64x2_extmul_low_u32x4(lhs4, rhs4),
            ),
        );
        let sum13 = u64x2_add(
            u64x2_add(
                u64x2_extmul_high_u32x4(lhs1, rhs1),
                u64x2_extmul_high_u32x4(lhs2, rhs2),
            ),
            u64x2_add(
                u64x2_extmul_high_u32x4(lhs3, rhs3),
                u64x2_extmul_high_u32x4(lhs4, rhs4),
            ),
        );
        let reduced02 = reduce_wide(sum02);
        let reduced13 = reduce_wide(sum13);
        // Interleaves `(c0, c2)` and `(c1, c3)`, in lanes 0 and 2 of each, into `(c0, c1, c2, c3)`.
        let reduced = u32x4_shuffle::<0, 4, 2, 6>(reduced02, reduced13);
        let mut out = [0u32; 4];
        // SAFETY: `out` is 16 bytes, the same size as a `v128`, and WebAssembly's `v128.store`
        // has no alignment requirement.
        unsafe { v128_store(out.as_mut_ptr().cast(), reduced) };
        out
    }

    /// WebAssembly SIMD version of [`super::scalar::kb_mul4x1`].
    ///
    /// The low and high widening multiplications by `rhs` yield the products of the first and
    /// last two coefficients; the two pairs are reduced with [`reduce_partially`], gathered back
    /// into `(c0, c1, c2, c3)`, and corrected with a single unsigned minimum.
    #[inline]
    pub(crate) fn kb_mul4x1(lhs: [u32; 4], rhs: u32) -> [u32; 4] {
        let [a0, a1, a2, a3] = lhs;
        let lhs = u32x4(a0, a1, a2, a3);
        let rhs = u32x4_splat(rhs);
        let low = reduce_partially(u64x2_extmul_low_u32x4(lhs, rhs));
        let high = reduce_partially(u64x2_extmul_high_u32x4(lhs, rhs));
        let gathered = u32x4_shuffle::<0, 2, 4, 6>(low, high);
        let reduced = u32x4_min(gathered, u32x4_sub(gathered, u32x4_splat(MODULUS)));
        let mut out = [0u32; 4];
        // SAFETY: `out` is 16 bytes, the same size as a `v128`, and WebAssembly's `v128.store`
        // has no alignment requirement.
        unsafe { v128_store(out.as_mut_ptr().cast(), reduced) };
        out
    }

    /// WebAssembly SIMD version of [`super::scalar::kb_add8`].
    ///
    /// Eight coefficients are two full `v128` lane vectors, so this is [`kb_add4`] on each half.
    #[inline]
    pub(crate) fn kb_add8(lhs: [u32; 8], rhs: [u32; 8]) -> [u32; 8] {
        let [a0, a1, a2, a3, a4, a5, a6, a7] = lhs;
        let [b0, b1, b2, b3, b4, b5, b6, b7] = rhs;
        let [c0, c1, c2, c3] = kb_add4([a0, a1, a2, a3], [b0, b1, b2, b3]);
        let [c4, c5, c6, c7] = kb_add4([a4, a5, a6, a7], [b4, b5, b6, b7]);
        [c0, c1, c2, c3, c4, c5, c6, c7]
    }

    /// WebAssembly SIMD version of [`super::scalar::kb_sub8`].
    ///
    /// Eight coefficients are two full `v128` lane vectors, so this is [`kb_sub4`] on each half.
    #[inline]
    pub(crate) fn kb_sub8(lhs: [u32; 8], rhs: [u32; 8]) -> [u32; 8] {
        let [a0, a1, a2, a3, a4, a5, a6, a7] = lhs;
        let [b0, b1, b2, b3, b4, b5, b6, b7] = rhs;
        let [c0, c1, c2, c3] = kb_sub4([a0, a1, a2, a3], [b0, b1, b2, b3]);
        let [c4, c5, c6, c7] = kb_sub4([a4, a5, a6, a7], [b4, b5, b6, b7]);
        [c0, c1, c2, c3, c4, c5, c6, c7]
    }

    /// WebAssembly SIMD version of [`super::scalar::kb_mul8`].
    ///
    /// The same Karatsuba over [`kb_mul4`], [`kb_add4`] and [`kb_sub4`], so the three KoalaBear^4
    /// multiplications and the combining steps all run the SIMD code above.
    #[inline]
    pub(crate) fn kb_mul8(lhs: [u32; 8], rhs: [u32; 8]) -> [u32; 8] {
        const { assert!(QUADRATIC_NON_RESIDUE == 3) }
        let [a0, a1, a2, a3, a4, a5, a6, a7] = lhs;
        let [b0, b1, b2, b3, b4, b5, b6, b7] = rhs;
        let ac = kb_mul4([a0, a1, a2, a3], [b0, b1, b2, b3]);
        let bd = kb_mul4([a4, a5, a6, a7], [b4, b5, b6, b7]);
        let s = kb_mul4(
            kb_add4([a0, a1, a2, a3], [a4, a5, a6, a7]),
            kb_add4([b0, b1, b2, b3], [b4, b5, b6, b7]),
        );
        let [ac0, ac1, ac2, ac3] = ac;
        let acy = [ac2, ac3, ac1, kb_add(kb_add(ac0, ac0), ac0)];
        let [c0, c1, c2, c3] = kb_sub4(kb_sub4(s, ac), bd);
        let [c4, c5, c6, c7] = kb_add4(bd, acy);
        [c0, c1, c2, c3, c4, c5, c6, c7]
    }

    /// WebAssembly SIMD version of [`super::scalar::kb_mul8x1`].
    ///
    /// Eight coefficients are two full `v128` lane vectors, so this is [`kb_mul4x1`] on each half.
    #[inline]
    pub(crate) fn kb_mul8x1(lhs: [u32; 8], rhs: u32) -> [u32; 8] {
        let [a0, a1, a2, a3, a4, a5, a6, a7] = lhs;
        let [c0, c1, c2, c3] = kb_mul4x1([a0, a1, a2, a3], rhs);
        let [c4, c5, c6, c7] = kb_mul4x1([a4, a5, a6, a7], rhs);
        [c0, c1, c2, c3, c4, c5, c6, c7]
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) use x86_64::{
    kb_add2, kb_add4, kb_add8, kb_mul2, kb_mul2x1, kb_mul4, kb_mul4x1, kb_mul8, kb_mul8x1, kb_sub2,
    kb_sub4, kb_sub8,
};

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(crate) use wasm32::{
    kb_add2, kb_add4, kb_add8, kb_mul2, kb_mul2x1, kb_mul4, kb_mul4x1, kb_mul8, kb_mul8x1, kb_sub2,
    kb_sub4, kb_sub8,
};

#[cfg(not(any(
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
)))]
pub(crate) use scalar::{
    kb_add2, kb_add4, kb_add8, kb_mul2, kb_mul2x1, kb_mul4, kb_mul4x1, kb_mul8, kb_mul8x1, kb_sub2,
    kb_sub4, kb_sub8,
};

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_WORDS: [u32; 8] = [
        0x1337, 0xdead, 0xbeef, 0xcafe, 0xbabe, 0xface, 0xfeed, 0xf00d,
    ];

    const TEST_SCALARS: [u32; 6] = [0, 1, 2, TEST_WORDS[7], MODULUS - 2, MODULUS - 1];

    const TEST_SCALARS_2: [[u32; 2]; 11] = [
        [0, 0],
        [0, 1],
        [0, 2],
        [1, 0],
        [1, 1],
        [1, 2],
        [TEST_WORDS[6], TEST_WORDS[7]],
        [MODULUS - 2, MODULUS - 2],
        [MODULUS - 2, MODULUS - 1],
        [MODULUS - 1, MODULUS - 2],
        [MODULUS - 1, MODULUS - 1],
    ];

    const TEST_SCALARS_4: [[u32; 4]; 6] = [
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 2],
        [TEST_WORDS[4], TEST_WORDS[5], TEST_WORDS[6], TEST_WORDS[7]],
        [MODULUS - 1, MODULUS - 1, MODULUS - 1, MODULUS - 2],
        [MODULUS - 1, MODULUS - 1, MODULUS - 1, MODULUS - 1],
    ];

    const TEST_SCALARS_8: [[u32; 8]; 6] = [
        [0; 8],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 2],
        TEST_WORDS,
        [
            MODULUS - 1,
            MODULUS - 1,
            MODULUS - 1,
            MODULUS - 1,
            MODULUS - 1,
            MODULUS - 1,
            MODULUS - 1,
            MODULUS - 2,
        ],
        [
            MODULUS - 1,
            MODULUS - 1,
            MODULUS - 1,
            MODULUS - 1,
            MODULUS - 1,
            MODULUS - 1,
            MODULUS - 1,
            MODULUS - 1,
        ],
    ];

    const MODULUS_64: u64 = MODULUS as u64;

    const fn reference_pow(base: u32, mut exponent: u32) -> u32 {
        let mut result = 1u64;
        let mut base = base as u64;
        while exponent > 0 {
            if exponent & 1 != 0 {
                result = (result * base) % MODULUS_64;
            }
            base = (base * base) % MODULUS_64;
            exponent >>= 1;
        }
        result as u32
    }

    fn reference_add(lhs: u32, rhs: u32) -> u32 {
        ((lhs as u64 + rhs as u64) % MODULUS_64) as u32
    }

    fn reference_sub(lhs: u32, rhs: u32) -> u32 {
        ((lhs as u64 + MODULUS_64 - rhs as u64) % MODULUS_64) as u32
    }

    fn reference_mul(lhs: u32, rhs: u32) -> u32 {
        ((lhs as u64) * (rhs as u64) % MODULUS_64) as u32
    }

    /// Canonical `(a * X + b) * (c * X + d)` with `X^2 = QUADRATIC_NON_RESIDUE`.
    fn reference_mul2(a: u32, b: u32, c: u32, d: u32) -> (u32, u32) {
        let hi = reference_add(reference_mul(a, d), reference_mul(b, c));
        let lo = reference_add(
            reference_mul(b, d),
            reference_mul(QUADRATIC_NON_RESIDUE, reference_mul(a, c)),
        );
        (hi, lo)
    }

    /// Canonical `(A * Y + B) * (C * Y + D)` with `Y^2 = X`.
    fn reference_mul4(lhs: [u32; 4], rhs: [u32; 4]) -> [u32; 4] {
        let [a, b, c, d] = lhs;
        let [e, f, g, h] = rhs;
        let ac = reference_mul2(a, b, e, f);
        let bd = reference_mul2(c, d, g, h);
        let ad = reference_mul2(a, b, g, h);
        let bc = reference_mul2(c, d, e, f);
        let acx = (ac.1, reference_mul(QUADRATIC_NON_RESIDUE, ac.0));
        [
            reference_add(ad.0, bc.0),
            reference_add(ad.1, bc.1),
            reference_add(bd.0, acx.0),
            reference_add(bd.1, acx.1),
        ]
    }

    /// Karatsuba over [`kb_mul2`]: an independent Montgomery implementation to cross-check
    /// [`kb_mul4`] against.
    fn karatsuba_mul4(lhs: [u32; 4], rhs: [u32; 4]) -> [u32; 4] {
        let [a, b, c, d] = lhs;
        let [e, f, g, h] = rhs;
        let [ac0, ac1] = kb_mul2([a, b], [e, f]);
        let [bd0, bd1] = kb_mul2([c, d], [g, h]);
        let [s0, s1] = kb_mul2([kb_add(a, c), kb_add(b, d)], [kb_add(e, g), kb_add(f, h)]);
        let ad_bc0 = kb_sub(kb_sub(s0, ac0), bd0);
        let ad_bc1 = kb_sub(kb_sub(s1, ac1), bd1);
        let acx0 = ac1;
        let acx1 = kb_add(kb_add(ac0, ac0), ac0);
        [ad_bc0, ad_bc1, kb_add(bd0, acx0), kb_add(bd1, acx1)]
    }

    /// Canonical `(A * Z + B) * (C * Z + D)` with `Z^2 = Y`.
    fn reference_mul8(lhs: [u32; 8], rhs: [u32; 8]) -> [u32; 8] {
        let [a0, a1, a2, a3, b0, b1, b2, b3] = lhs;
        let [c0, c1, c2, c3, d0, d1, d2, d3] = rhs;
        let (a, b) = ([a0, a1, a2, a3], [b0, b1, b2, b3]);
        let (c, d) = ([c0, c1, c2, c3], [d0, d1, d2, d3]);
        let ac = reference_mul4(a, c);
        let bd = reference_mul4(b, d);
        let ad = reference_mul4(a, d);
        let bc = reference_mul4(b, c);
        let acy = [
            ac[2],
            ac[3],
            ac[1],
            reference_mul(QUADRATIC_NON_RESIDUE, ac[0]),
        ];
        [
            reference_add(ad[0], bc[0]),
            reference_add(ad[1], bc[1]),
            reference_add(ad[2], bc[2]),
            reference_add(ad[3], bc[3]),
            reference_add(bd[0], acy[0]),
            reference_add(bd[1], acy[1]),
            reference_add(bd[2], acy[2]),
            reference_add(bd[3], acy[3]),
        ]
    }

    /// Schoolbook over [`kb_mul4`]: an independent Montgomery implementation to cross-check
    /// [`kb_mul8`] against.
    fn schoolbook_mul8(lhs: [u32; 8], rhs: [u32; 8]) -> [u32; 8] {
        let [a0, a1, a2, a3, b0, b1, b2, b3] = lhs;
        let [c0, c1, c2, c3, d0, d1, d2, d3] = rhs;
        let [ac0, ac1, ac2, ac3] = kb_mul4([a0, a1, a2, a3], [c0, c1, c2, c3]);
        let [bd0, bd1, bd2, bd3] = kb_mul4([b0, b1, b2, b3], [d0, d1, d2, d3]);
        let [ad0, ad1, ad2, ad3] = kb_mul4([a0, a1, a2, a3], [d0, d1, d2, d3]);
        let [bc0, bc1, bc2, bc3] = kb_mul4([b0, b1, b2, b3], [c0, c1, c2, c3]);
        let acy3 = kb_add(kb_add(ac0, ac0), ac0);
        [
            kb_add(ad0, bc0),
            kb_add(ad1, bc1),
            kb_add(ad2, bc2),
            kb_add(ad3, bc3),
            kb_add(bd0, ac2),
            kb_add(bd1, ac3),
            kb_add(bd2, ac1),
            kb_add(bd3, acy3),
        ]
    }

    #[test]
    fn test_modulus() {
        assert_eq!(MODULUS, (1 << 31) - (1 << 24) + 1);
    }

    #[test]
    fn test_r() {
        assert_eq!(R as u64, (1u64 << 32) % MODULUS_64);
        assert_eq!(R2 as u64, (R as u64) * (R as u64) % MODULUS_64);
    }

    #[test]
    fn test_p_inv() {
        assert_eq!(MODULUS.wrapping_mul(P_INV), u32::MAX);
    }

    #[test]
    fn test_quadratic_non_residue() {
        assert_eq!(
            reference_pow(QUADRATIC_NON_RESIDUE, (MODULUS - 1) / 2),
            MODULUS - 1
        );
    }

    fn test_add_impl(lhs: u32, rhs: u32) {
        assert_eq!(kb_add(lhs, rhs), reference_add(lhs, rhs));
        assert_eq!(kb_add(rhs, lhs), reference_add(lhs, rhs));
    }

    #[test]
    fn test_add() {
        for i in 0..TEST_SCALARS.len() {
            for j in i..TEST_SCALARS.len() {
                test_add_impl(TEST_SCALARS[i], TEST_SCALARS[j]);
            }
        }
    }

    fn test_sub_impl(lhs: u32, rhs: u32) {
        assert_eq!(kb_sub(lhs, rhs), reference_sub(lhs, rhs));
        assert_eq!(kb_sub(rhs, lhs), reference_sub(rhs, lhs));
        assert_eq!(kb_add(kb_sub(lhs, rhs), rhs), lhs);
    }

    #[test]
    fn test_sub() {
        for i in 0..TEST_SCALARS.len() {
            for j in i..TEST_SCALARS.len() {
                test_sub_impl(TEST_SCALARS[i], TEST_SCALARS[j]);
            }
        }
    }

    fn test_mul_impl(lhs: u32, rhs: u32) {
        let (lhs_montgomery, rhs_montgomery) = (kb_to_montgomery(lhs), kb_to_montgomery(rhs));
        let product = kb_mul(lhs_montgomery, rhs_montgomery);
        assert!(product < MODULUS);
        assert_eq!(kb_from_montgomery(product), reference_mul(lhs, rhs));
        assert_eq!(kb_mul(rhs_montgomery, lhs_montgomery), product);
    }

    #[test]
    fn test_mul() {
        for i in 0..TEST_SCALARS.len() {
            for j in i..TEST_SCALARS.len() {
                test_mul_impl(TEST_SCALARS[i], TEST_SCALARS[j]);
            }
        }
    }

    fn test_montgomery_conversion_impl(value: u32) {
        let montgomery = kb_to_montgomery(value);
        assert!(montgomery < MODULUS);
        assert_eq!(montgomery as u64, (value as u64) * (R as u64) % MODULUS_64);
        assert_eq!(kb_from_montgomery(montgomery), value % MODULUS);
    }

    #[test]
    fn test_montgomery_conversion() {
        test_montgomery_conversion_impl(0);
        test_montgomery_conversion_impl(1);
        test_montgomery_conversion_impl(2);
        test_montgomery_conversion_impl(0x13371337);
        test_montgomery_conversion_impl(MODULUS - 2);
        test_montgomery_conversion_impl(MODULUS - 1);
        test_montgomery_conversion_impl(MODULUS);
        test_montgomery_conversion_impl(MODULUS + 1);
        test_montgomery_conversion_impl(MODULUS + 2);
        test_montgomery_conversion_impl(u32::MAX - 1);
        test_montgomery_conversion_impl(u32::MAX);
    }

    fn test_scalar_add2_impl(lhs: [u32; 2], rhs: [u32; 2]) {
        let sum = [reference_add(lhs[0], rhs[0]), reference_add(lhs[1], rhs[1])];
        assert_eq!(scalar::kb_add2(lhs, rhs), sum);
        assert_eq!(scalar::kb_add2(rhs, lhs), sum);
    }

    #[cfg(target_arch = "x86_64")]
    fn test_add2_impl(lhs: [u32; 2], rhs: [u32; 2]) {
        let sum = [reference_add(lhs[0], rhs[0]), reference_add(lhs[1], rhs[1])];
        assert_eq!(x86_64::kb_add2(lhs, rhs), sum);
        assert_eq!(x86_64::kb_add2(rhs, lhs), sum);
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    fn test_add2_impl(lhs: [u32; 2], rhs: [u32; 2]) {
        let sum = [reference_add(lhs[0], rhs[0]), reference_add(lhs[1], rhs[1])];
        assert_eq!(wasm32::kb_add2(lhs, rhs), sum);
        assert_eq!(wasm32::kb_add2(rhs, lhs), sum);
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    fn test_add2_impl(_lhs: [u32; 2], _rhs: [u32; 2]) {}

    #[test]
    fn test_add2() {
        for i in 0..TEST_SCALARS_2.len() {
            for j in i..TEST_SCALARS_2.len() {
                test_scalar_add2_impl(TEST_SCALARS_2[i], TEST_SCALARS_2[j]);
                test_add2_impl(TEST_SCALARS_2[i], TEST_SCALARS_2[j]);
            }
        }
    }

    fn test_scalar_sub2_impl(lhs: [u32; 2], rhs: [u32; 2]) {
        let difference = [reference_sub(lhs[0], rhs[0]), reference_sub(lhs[1], rhs[1])];
        assert_eq!(scalar::kb_sub2(lhs, rhs), difference);
        assert_eq!(scalar::kb_add2(difference, rhs), lhs);
    }

    #[cfg(target_arch = "x86_64")]
    fn test_sub2_impl(lhs: [u32; 2], rhs: [u32; 2]) {
        let difference = [reference_sub(lhs[0], rhs[0]), reference_sub(lhs[1], rhs[1])];
        assert_eq!(x86_64::kb_sub2(lhs, rhs), difference);
        assert_eq!(x86_64::kb_add2(difference, rhs), lhs);
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    fn test_sub2_impl(lhs: [u32; 2], rhs: [u32; 2]) {
        let difference = [reference_sub(lhs[0], rhs[0]), reference_sub(lhs[1], rhs[1])];
        assert_eq!(wasm32::kb_sub2(lhs, rhs), difference);
        assert_eq!(wasm32::kb_add2(difference, rhs), lhs);
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    fn test_sub2_impl(_lhs: [u32; 2], _rhs: [u32; 2]) {}

    #[test]
    fn test_sub2() {
        for i in 0..TEST_SCALARS_2.len() {
            for j in i..TEST_SCALARS_2.len() {
                test_scalar_sub2_impl(TEST_SCALARS_2[i], TEST_SCALARS_2[j]);
                test_sub2_impl(TEST_SCALARS_2[i], TEST_SCALARS_2[j]);
            }
        }
    }

    fn test_scalar_add4_impl(lhs: [u32; 4], rhs: [u32; 4]) {
        let sum = [
            reference_add(lhs[0], rhs[0]),
            reference_add(lhs[1], rhs[1]),
            reference_add(lhs[2], rhs[2]),
            reference_add(lhs[3], rhs[3]),
        ];
        assert_eq!(scalar::kb_add4(lhs, rhs), sum);
        assert_eq!(scalar::kb_add4(rhs, lhs), sum);
    }

    #[cfg(target_arch = "x86_64")]
    fn test_add4_impl(lhs: [u32; 4], rhs: [u32; 4]) {
        let sum = [
            reference_add(lhs[0], rhs[0]),
            reference_add(lhs[1], rhs[1]),
            reference_add(lhs[2], rhs[2]),
            reference_add(lhs[3], rhs[3]),
        ];
        assert_eq!(x86_64::kb_add4(lhs, rhs), sum);
        assert_eq!(x86_64::kb_add4(rhs, lhs), sum);
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    fn test_add4_impl(lhs: [u32; 4], rhs: [u32; 4]) {
        let sum = [
            reference_add(lhs[0], rhs[0]),
            reference_add(lhs[1], rhs[1]),
            reference_add(lhs[2], rhs[2]),
            reference_add(lhs[3], rhs[3]),
        ];
        assert_eq!(wasm32::kb_add4(lhs, rhs), sum);
        assert_eq!(wasm32::kb_add4(rhs, lhs), sum);
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    fn test_add4_impl(_lhs: [u32; 4], _rhs: [u32; 4]) {}

    #[test]
    fn test_add4() {
        for i in 0..TEST_SCALARS_4.len() {
            for j in i..TEST_SCALARS_4.len() {
                test_scalar_add4_impl(TEST_SCALARS_4[i], TEST_SCALARS_4[j]);
                test_add4_impl(TEST_SCALARS_4[i], TEST_SCALARS_4[j]);
            }
        }
    }

    fn test_scalar_sub4_impl(lhs: [u32; 4], rhs: [u32; 4]) {
        let difference = [
            reference_sub(lhs[0], rhs[0]),
            reference_sub(lhs[1], rhs[1]),
            reference_sub(lhs[2], rhs[2]),
            reference_sub(lhs[3], rhs[3]),
        ];
        assert_eq!(scalar::kb_sub4(lhs, rhs), difference);
        assert_eq!(scalar::kb_add4(difference, rhs), lhs);
    }

    #[cfg(target_arch = "x86_64")]
    fn test_sub4_impl(lhs: [u32; 4], rhs: [u32; 4]) {
        let difference = [
            reference_sub(lhs[0], rhs[0]),
            reference_sub(lhs[1], rhs[1]),
            reference_sub(lhs[2], rhs[2]),
            reference_sub(lhs[3], rhs[3]),
        ];
        assert_eq!(x86_64::kb_sub4(lhs, rhs), difference);
        assert_eq!(x86_64::kb_add4(difference, rhs), lhs);
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    fn test_sub4_impl(lhs: [u32; 4], rhs: [u32; 4]) {
        let difference = [
            reference_sub(lhs[0], rhs[0]),
            reference_sub(lhs[1], rhs[1]),
            reference_sub(lhs[2], rhs[2]),
            reference_sub(lhs[3], rhs[3]),
        ];
        assert_eq!(wasm32::kb_sub4(lhs, rhs), difference);
        assert_eq!(wasm32::kb_add4(difference, rhs), lhs);
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    fn test_sub4_impl(_lhs: [u32; 4], _rhs: [u32; 4]) {}

    #[test]
    fn test_sub4() {
        for i in 0..TEST_SCALARS_4.len() {
            for j in i..TEST_SCALARS_4.len() {
                test_scalar_sub4_impl(TEST_SCALARS_4[i], TEST_SCALARS_4[j]);
                test_sub4_impl(TEST_SCALARS_4[i], TEST_SCALARS_4[j]);
            }
        }
    }

    fn test_scalar_add8_impl(lhs: [u32; 8], rhs: [u32; 8]) {
        let sum = [
            reference_add(lhs[0], rhs[0]),
            reference_add(lhs[1], rhs[1]),
            reference_add(lhs[2], rhs[2]),
            reference_add(lhs[3], rhs[3]),
            reference_add(lhs[4], rhs[4]),
            reference_add(lhs[5], rhs[5]),
            reference_add(lhs[6], rhs[6]),
            reference_add(lhs[7], rhs[7]),
        ];
        assert_eq!(scalar::kb_add8(lhs, rhs), sum);
        assert_eq!(scalar::kb_add8(rhs, lhs), sum);
    }

    #[cfg(target_arch = "x86_64")]
    fn test_add8_impl(lhs: [u32; 8], rhs: [u32; 8]) {
        let sum = [
            reference_add(lhs[0], rhs[0]),
            reference_add(lhs[1], rhs[1]),
            reference_add(lhs[2], rhs[2]),
            reference_add(lhs[3], rhs[3]),
            reference_add(lhs[4], rhs[4]),
            reference_add(lhs[5], rhs[5]),
            reference_add(lhs[6], rhs[6]),
            reference_add(lhs[7], rhs[7]),
        ];
        assert_eq!(x86_64::kb_add8(lhs, rhs), sum);
        assert_eq!(x86_64::kb_add8(rhs, lhs), sum);
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    fn test_add8_impl(lhs: [u32; 8], rhs: [u32; 8]) {
        let sum = [
            reference_add(lhs[0], rhs[0]),
            reference_add(lhs[1], rhs[1]),
            reference_add(lhs[2], rhs[2]),
            reference_add(lhs[3], rhs[3]),
            reference_add(lhs[4], rhs[4]),
            reference_add(lhs[5], rhs[5]),
            reference_add(lhs[6], rhs[6]),
            reference_add(lhs[7], rhs[7]),
        ];
        assert_eq!(wasm32::kb_add8(lhs, rhs), sum);
        assert_eq!(wasm32::kb_add8(rhs, lhs), sum);
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    fn test_add8_impl(_lhs: [u32; 8], _rhs: [u32; 8]) {}

    #[test]
    fn test_add8() {
        for i in 0..TEST_SCALARS_8.len() {
            for j in i..TEST_SCALARS_8.len() {
                test_scalar_add8_impl(TEST_SCALARS_8[i], TEST_SCALARS_8[j]);
                test_add8_impl(TEST_SCALARS_8[i], TEST_SCALARS_8[j]);
            }
        }
    }

    fn test_scalar_sub8_impl(lhs: [u32; 8], rhs: [u32; 8]) {
        let difference = [
            reference_sub(lhs[0], rhs[0]),
            reference_sub(lhs[1], rhs[1]),
            reference_sub(lhs[2], rhs[2]),
            reference_sub(lhs[3], rhs[3]),
            reference_sub(lhs[4], rhs[4]),
            reference_sub(lhs[5], rhs[5]),
            reference_sub(lhs[6], rhs[6]),
            reference_sub(lhs[7], rhs[7]),
        ];
        assert_eq!(scalar::kb_sub8(lhs, rhs), difference);
        assert_eq!(scalar::kb_add8(difference, rhs), lhs);
    }

    #[cfg(target_arch = "x86_64")]
    fn test_sub8_impl(lhs: [u32; 8], rhs: [u32; 8]) {
        let difference = [
            reference_sub(lhs[0], rhs[0]),
            reference_sub(lhs[1], rhs[1]),
            reference_sub(lhs[2], rhs[2]),
            reference_sub(lhs[3], rhs[3]),
            reference_sub(lhs[4], rhs[4]),
            reference_sub(lhs[5], rhs[5]),
            reference_sub(lhs[6], rhs[6]),
            reference_sub(lhs[7], rhs[7]),
        ];
        assert_eq!(x86_64::kb_sub8(lhs, rhs), difference);
        assert_eq!(x86_64::kb_add8(difference, rhs), lhs);
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    fn test_sub8_impl(lhs: [u32; 8], rhs: [u32; 8]) {
        let difference = [
            reference_sub(lhs[0], rhs[0]),
            reference_sub(lhs[1], rhs[1]),
            reference_sub(lhs[2], rhs[2]),
            reference_sub(lhs[3], rhs[3]),
            reference_sub(lhs[4], rhs[4]),
            reference_sub(lhs[5], rhs[5]),
            reference_sub(lhs[6], rhs[6]),
            reference_sub(lhs[7], rhs[7]),
        ];
        assert_eq!(wasm32::kb_sub8(lhs, rhs), difference);
        assert_eq!(wasm32::kb_add8(difference, rhs), lhs);
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    fn test_sub8_impl(_lhs: [u32; 8], _rhs: [u32; 8]) {}

    #[test]
    fn test_sub8() {
        for i in 0..TEST_SCALARS_8.len() {
            for j in i..TEST_SCALARS_8.len() {
                test_scalar_sub8_impl(TEST_SCALARS_8[i], TEST_SCALARS_8[j]);
                test_sub8_impl(TEST_SCALARS_8[i], TEST_SCALARS_8[j]);
            }
        }
    }

    fn test_scalar_mul2_impl(lhs: [u32; 2], rhs: [u32; 2]) {
        let lhs_montgomery = lhs.map(kb_to_montgomery);
        let rhs_montgomery = rhs.map(kb_to_montgomery);
        let product = scalar::kb_mul2(lhs_montgomery, rhs_montgomery);
        assert!(product.iter().all(|&coefficient| coefficient < MODULUS));
        let (hi, lo) = reference_mul2(lhs[0], lhs[1], rhs[0], rhs[1]);
        assert_eq!(product.map(kb_from_montgomery), [hi, lo]);
        assert_eq!(scalar::kb_mul2(rhs_montgomery, lhs_montgomery), product);
    }

    #[cfg(target_arch = "x86_64")]
    fn test_mul2_impl(lhs: [u32; 2], rhs: [u32; 2]) {
        let lhs_montgomery = lhs.map(kb_to_montgomery);
        let rhs_montgomery = rhs.map(kb_to_montgomery);
        let product = x86_64::kb_mul2(lhs_montgomery, rhs_montgomery);
        assert!(product.iter().all(|&coefficient| coefficient < MODULUS));
        let (hi, lo) = reference_mul2(lhs[0], lhs[1], rhs[0], rhs[1]);
        assert_eq!(product.map(kb_from_montgomery), [hi, lo]);
        assert_eq!(x86_64::kb_mul2(rhs_montgomery, lhs_montgomery), product);
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    fn test_mul2_impl(lhs: [u32; 2], rhs: [u32; 2]) {
        let lhs_montgomery = lhs.map(kb_to_montgomery);
        let rhs_montgomery = rhs.map(kb_to_montgomery);
        let product = wasm32::kb_mul2(lhs_montgomery, rhs_montgomery);
        assert!(product.iter().all(|&coefficient| coefficient < MODULUS));
        let (hi, lo) = reference_mul2(lhs[0], lhs[1], rhs[0], rhs[1]);
        assert_eq!(product.map(kb_from_montgomery), [hi, lo]);
        assert_eq!(wasm32::kb_mul2(rhs_montgomery, lhs_montgomery), product);
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    fn test_mul2_impl(_lhs: [u32; 2], _rhs: [u32; 2]) {}

    #[test]
    fn test_mul2() {
        for i in 0..TEST_SCALARS_2.len() {
            for j in i..TEST_SCALARS_2.len() {
                test_scalar_mul2_impl(TEST_SCALARS_2[i], TEST_SCALARS_2[j]);
                test_mul2_impl(TEST_SCALARS_2[i], TEST_SCALARS_2[j]);
            }
        }
    }

    #[test]
    fn test_mul2_tower_identity() {
        let x = [1, 0].map(kb_to_montgomery);
        assert_eq!(
            kb_mul2(x, x),
            [0, QUADRATIC_NON_RESIDUE].map(kb_to_montgomery)
        );
    }

    fn test_scalar_mul2x1_impl(lhs: [u32; 2], rhs: u32) {
        let lhs_montgomery = lhs.map(kb_to_montgomery);
        let rhs_montgomery = kb_to_montgomery(rhs);
        let product = scalar::kb_mul2x1(lhs_montgomery, rhs_montgomery);
        assert!(product.iter().all(|&coefficient| coefficient < MODULUS));
        assert_eq!(
            product.map(kb_from_montgomery),
            lhs.map(|coefficient| reference_mul(coefficient, rhs))
        );
        assert_eq!(
            scalar::kb_mul2(lhs_montgomery, [0, rhs_montgomery]),
            product
        );
    }

    #[cfg(target_arch = "x86_64")]
    fn test_mul2x1_impl(lhs: [u32; 2], rhs: u32) {
        let lhs_montgomery = lhs.map(kb_to_montgomery);
        let rhs_montgomery = kb_to_montgomery(rhs);
        let product = x86_64::kb_mul2x1(lhs_montgomery, rhs_montgomery);
        assert!(product.iter().all(|&coefficient| coefficient < MODULUS));
        assert_eq!(
            product.map(kb_from_montgomery),
            lhs.map(|coefficient| reference_mul(coefficient, rhs))
        );
        assert_eq!(
            x86_64::kb_mul2(lhs_montgomery, [0, rhs_montgomery]),
            product
        );
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    fn test_mul2x1_impl(lhs: [u32; 2], rhs: u32) {
        let lhs_montgomery = lhs.map(kb_to_montgomery);
        let rhs_montgomery = kb_to_montgomery(rhs);
        let product = wasm32::kb_mul2x1(lhs_montgomery, rhs_montgomery);
        assert!(product.iter().all(|&coefficient| coefficient < MODULUS));
        assert_eq!(
            product.map(kb_from_montgomery),
            lhs.map(|coefficient| reference_mul(coefficient, rhs))
        );
        assert_eq!(
            wasm32::kb_mul2(lhs_montgomery, [0, rhs_montgomery]),
            product
        );
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    fn test_mul2x1_impl(_lhs: [u32; 2], _rhs: u32) {}

    #[test]
    fn test_mul2x1() {
        for lhs in TEST_SCALARS_2 {
            for rhs in TEST_SCALARS {
                test_scalar_mul2x1_impl(lhs, rhs);
                test_mul2x1_impl(lhs, rhs);
            }
        }
    }

    /// Checks [`scalar::kb_mul4`] on `lhs` and `rhs` against the reference multiplication, against
    /// Karatsuba over `kb_mul2`, and against the multiplicative identity.
    fn test_scalar_mul4_impl(lhs: [u32; 4], rhs: [u32; 4]) {
        let one = kb_to_montgomery(1);
        let lhs_montgomery = lhs.map(kb_to_montgomery);
        let rhs_montgomery = rhs.map(kb_to_montgomery);
        let product = scalar::kb_mul4(lhs_montgomery, rhs_montgomery);
        assert!(product.iter().all(|&coefficient| coefficient < MODULUS));
        assert_eq!(product.map(kb_from_montgomery), reference_mul4(lhs, rhs));
        assert_eq!(product, karatsuba_mul4(lhs_montgomery, rhs_montgomery));
        assert_eq!(scalar::kb_mul4(rhs_montgomery, lhs_montgomery), product);
        assert_eq!(
            scalar::kb_mul4(lhs_montgomery, [0, 0, 0, one]),
            lhs_montgomery
        );
    }

    #[cfg(target_arch = "x86_64")]
    fn test_mul4_impl(lhs: [u32; 4], rhs: [u32; 4]) {
        let one = kb_to_montgomery(1);
        let lhs_montgomery = lhs.map(kb_to_montgomery);
        let rhs_montgomery = rhs.map(kb_to_montgomery);
        let product = x86_64::kb_mul4(lhs_montgomery, rhs_montgomery);
        assert!(product.iter().all(|&coefficient| coefficient < MODULUS));
        assert_eq!(product.map(kb_from_montgomery), reference_mul4(lhs, rhs));
        assert_eq!(product, karatsuba_mul4(lhs_montgomery, rhs_montgomery));
        assert_eq!(x86_64::kb_mul4(rhs_montgomery, lhs_montgomery), product);
        assert_eq!(
            x86_64::kb_mul4(lhs_montgomery, [0, 0, 0, one]),
            lhs_montgomery
        );
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    fn test_mul4_impl(lhs: [u32; 4], rhs: [u32; 4]) {
        let one = kb_to_montgomery(1);
        let lhs_montgomery = lhs.map(kb_to_montgomery);
        let rhs_montgomery = rhs.map(kb_to_montgomery);
        let product = wasm32::kb_mul4(lhs_montgomery, rhs_montgomery);
        assert!(product.iter().all(|&coefficient| coefficient < MODULUS));
        assert_eq!(product.map(kb_from_montgomery), reference_mul4(lhs, rhs));
        assert_eq!(product, karatsuba_mul4(lhs_montgomery, rhs_montgomery));
        assert_eq!(wasm32::kb_mul4(rhs_montgomery, lhs_montgomery), product);
        assert_eq!(
            wasm32::kb_mul4(lhs_montgomery, [0, 0, 0, one]),
            lhs_montgomery
        );
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    fn test_mul4_impl(_lhs: [u32; 4], _rhs: [u32; 4]) {}

    #[test]
    fn test_mul4() {
        for i in 0..TEST_SCALARS_4.len() {
            for j in i..TEST_SCALARS_4.len() {
                test_scalar_mul4_impl(TEST_SCALARS_4[i], TEST_SCALARS_4[j]);
                test_mul4_impl(TEST_SCALARS_4[i], TEST_SCALARS_4[j]);
            }
        }
    }

    #[test]
    fn test_mul4_tower_identity() {
        let y = [0, 1, 0, 0].map(kb_to_montgomery);
        let x = [0, 0, 1, 0].map(kb_to_montgomery);
        assert_eq!(kb_mul4(y, y), x);
        assert_eq!(
            kb_mul4(x, x),
            [0, 0, 0, QUADRATIC_NON_RESIDUE].map(kb_to_montgomery)
        );
    }

    fn test_scalar_mul4x1_impl(lhs: [u32; 4], rhs: u32) {
        let lhs_montgomery = lhs.map(kb_to_montgomery);
        let rhs_montgomery = kb_to_montgomery(rhs);
        let product = scalar::kb_mul4x1(lhs_montgomery, rhs_montgomery);
        assert!(product.iter().all(|&coefficient| coefficient < MODULUS));
        assert_eq!(
            product.map(kb_from_montgomery),
            lhs.map(|coefficient| reference_mul(coefficient, rhs))
        );
        assert_eq!(
            scalar::kb_mul4(lhs_montgomery, [0, 0, 0, rhs_montgomery]),
            product
        );
    }

    #[cfg(target_arch = "x86_64")]
    fn test_mul4x1_impl(lhs: [u32; 4], rhs: u32) {
        let lhs_montgomery = lhs.map(kb_to_montgomery);
        let rhs_montgomery = kb_to_montgomery(rhs);
        let product = x86_64::kb_mul4x1(lhs_montgomery, rhs_montgomery);
        assert!(product.iter().all(|&coefficient| coefficient < MODULUS));
        assert_eq!(
            product.map(kb_from_montgomery),
            lhs.map(|coefficient| reference_mul(coefficient, rhs))
        );
        assert_eq!(
            x86_64::kb_mul4(lhs_montgomery, [0, 0, 0, rhs_montgomery]),
            product
        );
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    fn test_mul4x1_impl(lhs: [u32; 4], rhs: u32) {
        let lhs_montgomery = lhs.map(kb_to_montgomery);
        let rhs_montgomery = kb_to_montgomery(rhs);
        let product = wasm32::kb_mul4x1(lhs_montgomery, rhs_montgomery);
        assert!(product.iter().all(|&coefficient| coefficient < MODULUS));
        assert_eq!(
            product.map(kb_from_montgomery),
            lhs.map(|coefficient| reference_mul(coefficient, rhs))
        );
        assert_eq!(
            wasm32::kb_mul4(lhs_montgomery, [0, 0, 0, rhs_montgomery]),
            product
        );
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    fn test_mul4x1_impl(_lhs: [u32; 4], _rhs: u32) {}

    #[test]
    fn test_mul4x1() {
        for lhs in TEST_SCALARS_4 {
            for rhs in TEST_SCALARS {
                test_scalar_mul4x1_impl(lhs, rhs);
                test_mul4x1_impl(lhs, rhs);
            }
        }
    }

    fn test_scalar_mul8_impl(lhs: [u32; 8], rhs: [u32; 8]) {
        let one = kb_to_montgomery(1);
        let lhs_montgomery = lhs.map(kb_to_montgomery);
        let rhs_montgomery = rhs.map(kb_to_montgomery);
        let product = scalar::kb_mul8(lhs_montgomery, rhs_montgomery);
        assert!(product.iter().all(|&coefficient| coefficient < MODULUS));
        assert_eq!(product.map(kb_from_montgomery), reference_mul8(lhs, rhs));
        assert_eq!(product, schoolbook_mul8(lhs_montgomery, rhs_montgomery));
        assert_eq!(scalar::kb_mul8(rhs_montgomery, lhs_montgomery), product);
        assert_eq!(
            scalar::kb_mul8(lhs_montgomery, [0, 0, 0, 0, 0, 0, 0, one]),
            lhs_montgomery
        );
    }

    #[cfg(target_arch = "x86_64")]
    fn test_mul8_impl(lhs: [u32; 8], rhs: [u32; 8]) {
        let one = kb_to_montgomery(1);
        let lhs_montgomery = lhs.map(kb_to_montgomery);
        let rhs_montgomery = rhs.map(kb_to_montgomery);
        let product = x86_64::kb_mul8(lhs_montgomery, rhs_montgomery);
        assert!(product.iter().all(|&coefficient| coefficient < MODULUS));
        assert_eq!(product.map(kb_from_montgomery), reference_mul8(lhs, rhs));
        assert_eq!(product, schoolbook_mul8(lhs_montgomery, rhs_montgomery));
        assert_eq!(x86_64::kb_mul8(rhs_montgomery, lhs_montgomery), product);
        assert_eq!(
            x86_64::kb_mul8(lhs_montgomery, [0, 0, 0, 0, 0, 0, 0, one]),
            lhs_montgomery
        );
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    fn test_mul8_impl(lhs: [u32; 8], rhs: [u32; 8]) {
        let one = kb_to_montgomery(1);
        let lhs_montgomery = lhs.map(kb_to_montgomery);
        let rhs_montgomery = rhs.map(kb_to_montgomery);
        let product = wasm32::kb_mul8(lhs_montgomery, rhs_montgomery);
        assert!(product.iter().all(|&coefficient| coefficient < MODULUS));
        assert_eq!(product.map(kb_from_montgomery), reference_mul8(lhs, rhs));
        assert_eq!(product, schoolbook_mul8(lhs_montgomery, rhs_montgomery));
        assert_eq!(wasm32::kb_mul8(rhs_montgomery, lhs_montgomery), product);
        assert_eq!(
            wasm32::kb_mul8(lhs_montgomery, [0, 0, 0, 0, 0, 0, 0, one]),
            lhs_montgomery
        );
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    fn test_mul8_impl(_lhs: [u32; 8], _rhs: [u32; 8]) {}

    #[test]
    fn test_mul8() {
        for i in 0..TEST_SCALARS_8.len() {
            for j in i..TEST_SCALARS_8.len() {
                test_scalar_mul8_impl(TEST_SCALARS_8[i], TEST_SCALARS_8[j]);
                test_mul8_impl(TEST_SCALARS_8[i], TEST_SCALARS_8[j]);
            }
        }
    }

    #[test]
    fn test_mul8_tower_identity() {
        let z = [0, 0, 0, 1, 0, 0, 0, 0].map(kb_to_montgomery);
        let y = [0, 0, 0, 0, 0, 1, 0, 0].map(kb_to_montgomery);
        let x = [0, 0, 0, 0, 0, 0, 1, 0].map(kb_to_montgomery);
        assert_eq!(kb_mul8(z, z), y);
        assert_eq!(kb_mul8(y, y), x);
        assert_eq!(
            kb_mul8(x, x),
            [0, 0, 0, 0, 0, 0, 0, QUADRATIC_NON_RESIDUE].map(kb_to_montgomery)
        );
    }

    fn test_scalar_mul8x1_impl(lhs: [u32; 8], rhs: u32) {
        let lhs_montgomery = lhs.map(kb_to_montgomery);
        let rhs_montgomery = kb_to_montgomery(rhs);
        let product = scalar::kb_mul8x1(lhs_montgomery, rhs_montgomery);
        assert!(product.iter().all(|&coefficient| coefficient < MODULUS));
        assert_eq!(
            product.map(kb_from_montgomery),
            lhs.map(|coefficient| reference_mul(coefficient, rhs))
        );
        assert_eq!(
            scalar::kb_mul8(lhs_montgomery, [0, 0, 0, 0, 0, 0, 0, rhs_montgomery]),
            product
        );
    }

    #[cfg(target_arch = "x86_64")]
    fn test_mul8x1_impl(lhs: [u32; 8], rhs: u32) {
        let lhs_montgomery = lhs.map(kb_to_montgomery);
        let rhs_montgomery = kb_to_montgomery(rhs);
        let product = x86_64::kb_mul8x1(lhs_montgomery, rhs_montgomery);
        assert!(product.iter().all(|&coefficient| coefficient < MODULUS));
        assert_eq!(
            product.map(kb_from_montgomery),
            lhs.map(|coefficient| reference_mul(coefficient, rhs))
        );
        assert_eq!(
            x86_64::kb_mul8(lhs_montgomery, [0, 0, 0, 0, 0, 0, 0, rhs_montgomery]),
            product
        );
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    fn test_mul8x1_impl(lhs: [u32; 8], rhs: u32) {
        let lhs_montgomery = lhs.map(kb_to_montgomery);
        let rhs_montgomery = kb_to_montgomery(rhs);
        let product = wasm32::kb_mul8x1(lhs_montgomery, rhs_montgomery);
        assert!(product.iter().all(|&coefficient| coefficient < MODULUS));
        assert_eq!(
            product.map(kb_from_montgomery),
            lhs.map(|coefficient| reference_mul(coefficient, rhs))
        );
        assert_eq!(
            wasm32::kb_mul8(lhs_montgomery, [0, 0, 0, 0, 0, 0, 0, rhs_montgomery]),
            product
        );
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    fn test_mul8x1_impl(_lhs: [u32; 8], _rhs: u32) {}

    #[test]
    fn test_mul8x1() {
        for lhs in TEST_SCALARS_8 {
            for rhs in TEST_SCALARS {
                test_scalar_mul8x1_impl(lhs, rhs);
                test_mul8x1_impl(lhs, rhs);
            }
        }
    }
}
