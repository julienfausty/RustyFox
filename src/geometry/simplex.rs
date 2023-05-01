use crate::geometry::geometry_traits::Geometry;

use num::integer::binomial;
use std::collections::HashMap;

/// Geometry structure representing a simplex
///
/// # Members
///
/// * dimension: the topological dimension of the simplex
/// * embedding_dimension: the dimension of the emedding coordinate space
/// * points: the coordinates of the points representing the simplex
/// * connectivity: a map between the (element_dim, description_dim) and the connectivity of the
/// element_dim elements in terms of description_dim elements with the number of description_dim
/// elements needed to describe one element_dim
///
/// # Notes
///
/// * This structure isn't meant to own its coordinate data. Should be a slice from data owned
/// somewhere else.
struct Simplex<'a, CoordType> {
    dimension: usize,
    embedding_dimension: usize,
    points: &'a [CoordType],
    connectivity: HashMap<(usize, usize), (usize, Vec<usize>)>
}

impl<'a, CoordType> Simplex<'a, CoordType> {
    pub fn new(dim: usize, embed_dim: usize, pnts: &'a [CoordType]) -> Result<Simplex<'a, CoordType>, &'static str> {
        if embed_dim < dim || pnts.len() != embed_dim * dim + 1 {
            return Err("Incorherence in the points or the dimensions given to construct the Simplex.");
        }
        let mut simplex = Simplex{ dimension : dim, embedding_dimension : embed_dim, points : pnts, connectivity : HashMap::new() };
        for ied in 0..dim {
            for ted in 0..dim {
                simplex.compute_adjacency(ied, ted);
            }
        }
        Ok(simplex)
    }
    pub fn compute_adjacency(&mut self, element_dim: usize, target_dim: usize) {
        // TODO
    }
}

impl<'a, CoordType> Geometry<CoordType, usize> for Simplex<'a, CoordType> {
    fn get_dimension(&self) -> usize {
        self.dimension
    }

    fn get_embedding_dimension(&self) -> usize {
        self.embedding_dimension
    }

    fn get_coordinates(&self) -> &'a [CoordType] {
        self.points
    }

    fn get_number_of_elements(&self, dimension: usize) -> usize {
        if dimension > self.dimension {
            return 0;
        }
        binomial(self.dimension + 1, dimension + 1)
    }

    fn get_connectivity(&self, target_dim: usize, element_dim: usize, element_index: usize) -> Result<&'a [usize], &'static str> {
        if target_dim > self.dimension || element_dim > self.dimension {
            return Err("Requested connectivity dimensions are over the topological dimension of the simplex");
        }
        match self.connectivity.get(&(element_dim, target_dim)) {
            None => Err("Requested connectivity is not available"),
            Some((stride, conn)) => Ok(&conn[element_index*stride..element_index*(stride+1)])
        }
    }
}
