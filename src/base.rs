use crate::helpers::MODULUS;

#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Scalar(u32);

impl Scalar {
    pub const fn from_const(value: u32) -> Self {
        assert!(value < MODULUS);
        Self(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // TODO
}
