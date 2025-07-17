use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::PyResult;

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
            if value >= self.values[worst_idx] {
                return None; // skip worse value
            }
            // Replace worst value at its position (no removal/shifting)
            self.values[worst_idx] = value;
            self.update_sorted_indices();
            return Some(worst_idx);
        }

        // Buffer is growing, add to next position
        let position = self.values.len();
        self.values.push(value);
        self.update_sorted_indices();
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

    fn update_sorted_indices(&mut self) {
        let mut idx: Vec<_> = (0..self.values.len()).collect();
        idx.sort_by(|&a, &b| self.values[a].partial_cmp(&self.values[b]).unwrap());
        self.sorted_indices = idx;
    }
}
