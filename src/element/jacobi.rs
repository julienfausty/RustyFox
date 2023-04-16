extern crate num;
use num::bigint::BigInt;
use num::bigint::ToBigInt;
use num::integer::binomial;
use num::ToPrimitive;

/// Structure representing a Jacobi polynomial
///
/// # Pseudo math
/// P_{n}^{\alpha, \beta}(x) = {
///   let J = Jacobi::new(n, \alpha, \beta);
///   J(x)
/// }
pub struct Jacobi {
    degree: usize,
    alpha: i32,
    beta: i32,
    coeffs: Vec<BigInt>,
    normalizer: f64,
}

impl Jacobi {
    /// Constructor
    ///
    /// # Arguments
    ///
    /// * `degree`: the polynomial degree
    /// * `alpha`: the first parameter of the Jacobi polynomial (must be > -1)
    /// * `beta`: the second parameter of the Jacobi polynomial (must be > -1)
    ///
    /// # Returns
    ///
    /// * An option either holding the structure or a None if the arguments passed to it were not
    /// acceptable
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

    /// Return the degree of the polynomial
    pub fn get_degree(&self) -> usize {
        self.degree
    }

    /// Return the alpha parameter of the polynomial
    pub fn get_alpha(&self) -> i32 {
        self.alpha
    }

    /// Return the beta parameter of the polynomial
    pub fn get_beta(&self) -> i32 {
        self.beta
    }

    /// Evaluate the Jacobi polynomial at x
    ///
    /// # Arguments
    ///
    /// * `x`: the real number to evaluate the polynomial at
    ///
    /// # Returns
    ///
    /// * the evaluation of P_{n}^{\alpha, \beta}(x)
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

#[cfg(test)]
mod tests {
    use super::Jacobi;

    const TOL: f64 = 1e-8;

    #[test]
    fn test_new() {
        let jac = Jacobi::new(1, 2, 3).unwrap();
        assert_eq!(jac.get_degree(), 1, "Incorrect degree in new");
        assert_eq!(jac.get_alpha(), 2, "Incorrect alpha in new");
        assert_eq!(jac.get_beta(), 3, "Incorrect beta in new");
    }

    #[test]
    fn test_none() {
        let jac = Jacobi::new(1, -2, 3);
        assert!(jac.is_none(), "Did not give None for alpha equal to -2");
        let jac = Jacobi::new(1, 2, -3);
        assert!(jac.is_none(), "Did not give None for bet equal to -3");
    }

    #[test]
    fn test_1_1_1() {
        let jac = Jacobi::new(1, 1, 1).unwrap();
        assert!((jac.evaluate(-1.0) + 2.0).abs() < TOL, "Incorrect -1 value");
        assert!((jac.evaluate(1.0) - 2.0).abs() < TOL, "Incorrect 1 value");
        assert!(jac.evaluate(0.0).abs() < TOL, "Incorrect 0 value");
    }

    #[test]
    fn test_2_1_1() {
        let jac = Jacobi::new(2, 1, 1).unwrap();
        assert!((jac.evaluate(-1.0) - 3.0).abs() < TOL, "Incorrect -1 value");
        assert!((jac.evaluate(1.0) - 3.0).abs() < TOL, "Incorrect 1 value");
        assert!(
            (jac.evaluate(-0.2) + 0.6).abs() < TOL,
            "Incorrect -0.2 value"
        );
        assert!((jac.evaluate(0.2) + 0.6) < TOL, "Incorrect 0.2 value");
    }

    #[test]
    fn test_2_2_1() {
        let jac = Jacobi::new(2, 2, 1).unwrap();
        assert!((jac.evaluate(-1.0) - 3.0).abs() < TOL, "Incorrect -1 value");
        assert!((jac.evaluate(1.0) - 6.0).abs() < TOL, "Incorrect 1 value");
        assert!(
            (jac.evaluate(-0.2) + 0.84).abs() < TOL,
            "Incorrect -0.2 value"
        );
        assert!((jac.evaluate(0.2) + 0.24) < TOL, "Incorrect 0.2 value");
    }

    #[test]
    fn test_2_1_2() {
        let jac = Jacobi::new(2, 1, 2).unwrap();
        assert!((jac.evaluate(-1.0) - 6.0).abs() < TOL, "Incorrect -1 value");
        assert!((jac.evaluate(1.0) - 3.0).abs() < TOL, "Incorrect 1 value");
        assert!(
            (jac.evaluate(-0.2) + 0.24).abs() < TOL,
            "Incorrect -0.2 value"
        );
        assert!((jac.evaluate(0.2) + 0.84) < TOL, "Incorrect 0.2 value");
    }

    #[test]
    fn test_3_2_3() {
        let jac = Jacobi::new(3, 2, 3).unwrap();
        assert!(
            (jac.evaluate(-1.0) + 20.0).abs() < TOL,
            "Incorrect -1 value"
        );
        assert!((jac.evaluate(1.0) - 10.0).abs() < TOL, "Incorrect 1 value");
        assert!(
            (jac.evaluate(-0.2) - 1.36).abs() < TOL,
            "Incorrect -0.2 value"
        );
        assert!((jac.evaluate(0.2) + 0.56) < TOL, "Incorrect 0.2 value");
        assert!((jac.evaluate(0.0) - 0.625) < TOL, "Incorrect 0.0 value");
    }

    #[test]
    fn test_6_3_1() {
        let jac = Jacobi::new(6, 3, 1).unwrap();
        assert!((jac.evaluate(-1.0) - 7.0).abs() < TOL, "Incorrect -1 value");
        assert!(
            (jac.evaluate(-0.7) + 0.3219348125) < TOL,
            "Incorrect -0.7 value"
        );
        assert!(
            (jac.evaluate(-0.2) + 0.756672).abs() < TOL,
            "Incorrect -0.2 value"
        );
        assert!((jac.evaluate(0.0) + 0.21875) < TOL, "Incorrect 0.0 value");
        assert!((jac.evaluate(0.2) - 1.189888) < TOL, "Incorrect 0.2 value");
        assert!(
            (jac.evaluate(0.7) + 1.588921688) < TOL,
            "Incorrect 0.2 value"
        );
        assert!((jac.evaluate(1.0) - 84.0).abs() < TOL, "Incorrect 1 value");
    }
}
