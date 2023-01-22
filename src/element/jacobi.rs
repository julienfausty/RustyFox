extern crate num;
use num::bigint::BigInt;
use num::bigint::ToBigInt;
use num::integer::binomial;
use num::ToPrimitive;

pub struct Jacobi {
    degree: usize,
    alpha: i32,
    beta: i32,
    coeffs: Vec<BigInt>,
    normalizer: f64,
}

impl Jacobi {
    pub fn new(degree: usize, alpha: i32, beta: i32) -> Option<Jacobi> {
        if alpha < -1 || beta < -1 {
            return None;
        }
        let coeff = |k: usize| -> BigInt {
            binomial(
                degree.to_bigint().unwrap() + alpha.to_bigint().unwrap(),
                k.to_bigint().unwrap(),
            ) * binomial(
                degree.to_bigint().unwrap() + beta.to_bigint().unwrap(),
                degree.to_bigint().unwrap() - k.to_bigint().unwrap(),
            )
        };
        let coeffs = (0..=degree).into_iter().map(coeff).collect();
        Some(Jacobi {
            degree: degree,
            alpha: alpha,
            beta: beta,
            coeffs: coeffs,
            normalizer: 1.0 / (2_i32.pow(degree.try_into().unwrap()) as f64),
        })
    }

    pub fn get_degree(&self) -> usize {
        self.degree
    }

    pub fn get_alpha(&self) -> i32 {
        self.alpha
    }

    pub fn get_beta(&self) -> i32 {
        self.beta
    }

    pub fn evaluate(&self, x: f64) -> f64 {
        let monome = |deg: usize| -> f64 {
            (x - 1.0).powf((self.degree - deg) as f64) * (x + 1.0).powf(deg as f64)
        };
        (0..=self.degree)
            .into_iter()
            .map(monome)
            .zip(self.coeffs.iter())
            .map(|(m, c)| m * (c.to_f64().unwrap()))
            .sum::<f64>()
            * self.normalizer
    }
}

#[cfg(tests)]
mod tests {
    use super::Jacobi;

    #[test]
    fn test_new() {
        let jac = Jacobi::new(1, 2, 3);
        assert_eq!(jac.get_degree(), 1, "Incorrect degree in new");
        assert_eq!(jac.get_alpha(), 2, "Incorrect alpha in new");
        assert_eq!(jac.get_beta(), 3, "Incorrect beta in new");
    }

    #[test]
    fn test_1_1_1() {
        let jac = Jacobi::new(1, 1, 1);
        assert_eq!(jac.evaluate(-1.0), -2.0, "Incorrect -1 value");
        assert_eq!(jac.evaluate(1.0), 2.0, "Incorrect 1 value");
        assert_eq!(jac.evaluate(0.0), 0.0, "Incorrect 0 value");
    }
}
