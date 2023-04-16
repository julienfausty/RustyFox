use ndarray::LinalgScalar;
use std::collections::HashMap;
use crate::element::element_traits::Element;

/// Computes a discrete matrix operator
///
///# Generics
///
/// * CoordType: represents the unit type of the base space
/// * DataType: the type of unit the data is encoded with
///
/// # Types
///
/// * ElementT: the type of element this operator should use to describe the cell
///
/// # Traits
///
/// * Fn(geometry , data): makes the operator callable
/// by passing the real geometry of a cell and the data associated to the cell in a HashMap. Should
/// return a flattened matrix.
///
/// # Explanation
/// Given the geometry of a cell and its associated data, compute a local matrix that embodies the
/// discretized operator
pub trait Operator<CoordType: LinalgScalar, DataType: LinalgScalar>:
    Fn(&[CoordType], &HashMap<String, &[DataType]>) -> Vec<DataType>
{
    type ElementT: Element<CoordType, DataType>;
}
