/// Provides coordinates and ordering describing a geometry
pub trait Geometry<CoordType> {
    /// Get the topological dimension of the geometry
    fn get_dimension(&self) -> usize;

    /// Get the number of elements of topological dimension  `dimension` in the geometry
    fn get_number_of_elements(&self, dimension: usize) -> usize;

    /// Get the embedding coordinates of the dimension 0 elements of the geometry.
    fn get_coordinates(&self);

    /// Get the connectivity of dimension `target_dimension` elements expressed in
    /// `represented_dimension` elements
    fn get_connectivity(&self, target_dimension: usize, represented_dimension: usize);
}
