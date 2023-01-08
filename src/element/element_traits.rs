use ndarray::Array;
use ndarray::ArrayView;
use ndarray::LinalgScalar;

/// Provides weights and points for discrete integration operations
///
/// # Generics
///
/// * CoordType: represents the unit type of the integration space
/// * DataType: the type of value one is integrating over (result and weights should be of same type)
///
/// # Explanation
///
/// Discrete integration is often expressed as a dot product between integration weights and values
/// of the integrand at specific points in the integration space. The combination of weights and
/// points is called a quadrature or cubature rule. An object implementing this trait should
/// implement something like a discrete cubature rule.
pub trait IntegrationRule<CoordType, DataType: LinalgScalar> {
    /// Get the underlying dimension of the point space
    fn get_dimension(&self) -> usize;

    /// Get the weights associated with this integration
    fn get_weights(&self) -> &[DataType];

    /// Get the points in AOS ordering
    fn get_points(&self) -> &[CoordType];

    /// Get the number of individual weights
    fn get_number_of_points(&self) -> usize;

    /// Integrate a range of values
    fn integrate(&self, values: &[DataType]) -> DataType {
        let values_map = ArrayView::from_shape(self.get_number_of_points(), values).unwrap();
        let weights_map =
            ArrayView::from_shape(self.get_number_of_points(), self.get_weights()).unwrap();
        weights_map.dot(&values_map)
    }
}

/// Provides basis shape functions for describing fields inside elements
///
/// # Generics
///
/// * CoordType: represents the unit type of the element space
/// * DataType: the type of unit the field is encoded with
///
/// # Explanation
///
/// In the finite element method, fields are encoded into arrays using element level basis
/// functions called shape functions. Objects implementing this trait should provide a distinct set
/// of shape functions through the ability to interpolate there values at a given coordinate inside
/// the element.
pub trait ShapeBasis<CoordType, DataType: LinalgScalar> {
    /// Get the underlying dimension of the space the shapes are defined on
    fn get_dimension(&self) -> usize;

    /// Get the number of DataType values to describe one shape function derivative value
    fn get_derivative_order(&self) -> usize {
        self.get_dimension()
    }

    /// Get the number of basis functions
    fn get_number_of_bases(&self) -> usize;

    /// Interpolate the basis functions at a given coordinate
    fn interpolate_basis(&self, coord: &[CoordType]) -> Vec<DataType>;

    /// Interpolate the basis functions' derivatives at a given coordinate
    fn interpolate_basis_derivative(&self, coord: &[CoordType]) -> Vec<DataType>;

    /// Interpolate the value of the function defined weighting each of basis function using
    /// the values argument at the point in the element defined by coord
    fn interpolate(&self, coord: &[CoordType], values: &[DataType]) -> DataType {
        let shapes = Array::from_vec(self.interpolate_basis(coord));
        let val_view = ArrayView::from(values);
        shapes.dot(&val_view)
    }

    ///Same as interpolate above but for the derivative of the function
    fn interpolate_derivative(&self, coord: &[CoordType], values: &[DataType]) -> Vec<DataType> {
        let shape_derives = Array::from_shape_vec(
            (self.get_number_of_bases(), self.get_derivative_order()),
            self.interpolate_basis_derivative(coord),
        )
        .unwrap();
        let values_view = ArrayView::from(values);
        values_view.dot(&shape_derives).to_vec()
    }
}

/// Provide all the basic building blocks for the reference element of the finite element method
///
/// # Generics
///
/// * CoordType: represents the unit type of the element space
/// * DataType: the type of unit the field is encoded with
///
/// # Types
///
/// * IntegratorT: the type of IntegrationRule used by the element
/// * ShapeBasisT: the type of shape basis used by the element
///
/// # Explanation
///
/// A classic finite element reference element is described by a basis of shape functions and an
/// integration rule. The geometry of the element is implicit in the shape basis and integration
/// rule. An object implementing this trait should provide access to an interpolator and integrator
/// as well as some precomputed values of the shape functions on the integration points.
pub trait Element<CoordType: LinalgScalar, DataType: LinalgScalar> {
    type IntegratorT: IntegrationRule<CoordType, DataType>;
    type ShapeBasisT: ShapeBasis<CoordType, DataType>;

    /// Get the integration rule
    fn get_integrator(&self) -> Self::IntegratorT;

    /// Get the shape basis
    fn get_shape_basis(&self) -> Self::ShapeBasisT;

    /// Get the values of the shape basis at the integration points in AOS ordering and shape
    /// `(number_integration_points, number_shape_basis_functions, shape_order)`
    fn get_shapes_for_integration(&self) -> &[DataType];

    /// Get the values of the derivatives of the shape basis at the integration points in AOS
    /// ordering and shape `(number_integration_points, number_shape_basis_functions, derivative_order)`
    fn get_shape_derivatives_for_integration(&self) -> &[DataType];

    /// Get the values of the jacobian matrices at the integration points for an element with geometry
    /// descibed by coords.
    ///
    /// Should return an AOS array of shape:
    /// `(number_integration_points, number_dimensions, derivative_order)`
    fn get_geometry_derivatives_for_integration(&self, coords: &[CoordType]) -> Vec<DataType>;

    /// Integrate a field inside the element with shape weights equal to values
    fn integrate(&self, values: &[DataType]) -> DataType {
        let nips = self.get_integrator().get_number_of_points();
        let nbases = self.get_shape_basis().get_number_of_bases();
        let shapes =
            ArrayView::from_shape((nips, nbases), self.get_shapes_for_integration()).unwrap();
        let ip_values = shapes.dot(&ArrayView::from_shape(nbases, values).unwrap());
        self.get_integrator()
            .integrate(ip_values.as_slice().unwrap())
    }
}
