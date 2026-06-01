use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::PyResult;
use std::cmp::Ordering;

#[pymodule]
fn buffer_core(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BufferCore>()?;
    Ok(())
}

#[pyclass]
pub struct BufferCore {
    values: Vec<Vec<f32>>,
    sorted_indices: Vec<usize>,
    max_size: usize,
    value_levels: usize,
}

impl BufferCore {
    fn cmp_f32(a: f32, b: f32) -> Ordering {
        match (a.is_nan(), b.is_nan()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            (false, false) => a.total_cmp(&b),
        }
    }

    fn cmp_value_vec(a: &[f32], b: &[f32]) -> Ordering {
        for (av, bv) in a.iter().zip(b.iter()) {
            let ord = Self::cmp_f32(*av, *bv);
            if ord != Ordering::Equal {
                return ord;
            }
        }
        a.len().cmp(&b.len())
    }

    fn sorted_insert_position(&self, idx: usize) -> usize {
        self.sorted_indices
            .binary_search_by(|&probe| Self::cmp_value_vec(&self.values[probe], &self.values[idx]))
            .unwrap_or_else(|pos| pos)
    }

    fn insert_sorted_index(&mut self, idx: usize) {
        let pos = self.sorted_insert_position(idx);
        self.sorted_indices.insert(pos, idx);
    }

    fn remove_sorted_index(&mut self, idx: usize) {
        if let Some(pos) = self.sorted_indices.iter().position(|&value| value == idx) {
            self.sorted_indices.remove(pos);
        }
    }
}

#[pymethods]
impl BufferCore {
    #[new]
    pub fn new(max_size: usize, value_levels: usize) -> Self {
        Self {
            values: Vec::with_capacity(max_size),
            sorted_indices: vec![],
            max_size,
            value_levels,
        }
    }

    /// Insert a new value vector. Returns the position where the value was placed.
    pub fn insert(&mut self, value: Vec<f32>) -> Option<usize> {
        assert_eq!(
            value.len(),
            self.value_levels,
            "Incorrect number of value levels"
        );

        if self.values.len() == self.max_size {
            let worst_idx = *self.sorted_indices.last().unwrap();
            if Self::cmp_value_vec(&value, &self.values[worst_idx]) != Ordering::Less {
                return None;
            }
            self.values[worst_idx] = value;
            self.remove_sorted_index(worst_idx);
            self.insert_sorted_index(worst_idx);
            return Some(worst_idx);
        }

        let position = self.values.len();
        self.values.push(value);
        self.insert_sorted_index(position);
        Some(position)
    }

    pub fn insert_many(&mut self, values: Vec<Vec<f32>>) {
        for value in values {
            self.insert(value);
        }
    }

    pub fn get_indices(&self) -> Vec<usize> {
        self.sorted_indices.clone()
    }

    pub fn get_mean(&self, level: usize) -> f32 {
        assert!(level < self.value_levels, "Invalid level index");
        let sum: f32 = self.values.iter().map(|v| v[level]).sum();
        sum / self.values.len().max(1) as f32
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn clear(&mut self) {
        self.values.clear();
        self.sorted_indices.clear();
    }

    pub fn get_value(&self, index: usize) -> Option<Vec<f32>> {
        if index < self.sorted_indices.len() {
            let actual_idx = self.sorted_indices[index];
            self.values.get(actual_idx).cloned()
        } else {
            None
        }
    }

    pub fn get_sorted_values(&self) -> Vec<Vec<f32>> {
        self.sorted_indices
            .iter()
            .map(|&i| self.values[i].clone())
            .collect()
    }
}
