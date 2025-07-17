use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::types::PyModule;
use pyo3::PyResult;

#[pymodule]
fn buffer_core(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BufferCore>()?;
    Ok(())
}

/// A hierarchically sorted buffer.
#[pyclass]
pub struct BufferCore {
    buffer: Vec<Py<PyAny>>, // opaque Python objects (e.g., tensors)
    values: Vec<Vec<f32>>,  // N-dimensional sort keys
    max_size: usize,
    value_levels: usize,
}

#[pymethods]
impl BufferCore {
    #[new]
    pub fn new(max_size: usize, value_levels: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(max_size),
            values: Vec::with_capacity(max_size),
            max_size,
            value_levels,
        }
    }

    pub fn insert(&mut self, _py: Python<'_>, value: Vec<f32>, tensor: Py<PyAny>) {
        assert_eq!(value.len(), self.value_levels, "Wrong number of levels");

        let idx = self
            .values
            .binary_search_by(|probe| {
                probe
                    .partial_cmp(&value)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or_else(|e| e);

        if self.values.len() == self.max_size {
            if idx >= self.max_size {
                return; // worse than all, skip
            }
            self.values.pop();
            self.buffer.pop();
        }

        self.values.insert(idx, value);
        self.buffer.insert(idx, tensor);
    }

    pub fn insert_many(&mut self, py: Python<'_>, values: Vec<Vec<f32>>, tensors: Vec<Py<PyAny>>) {
        assert_eq!(values.len(), tensors.len(), "Mismatched inputs");

        // Zip and sort before merging
        let mut zipped: Vec<_> = values.into_iter().zip(tensors.into_iter()).collect();
        zipped.sort_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        for (v, t) in zipped {
            self.insert(py, v, t);
        }
    }

    pub fn get_mean(&self, level: usize) -> f32 {
        assert!(level < self.value_levels);
        let sum: f32 = self.values.iter().map(|v| v[level]).sum();
        sum / self.values.len().max(1) as f32
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn clear(&mut self) {
        self.values.clear();
        self.buffer.clear();
    }

    pub fn get_tensor(&self, index: usize) -> Option<&Py<PyAny>> {
        self.buffer.get(index)
    }

    pub fn get_value(&self, index: usize) -> Option<Vec<f32>> {
        self.values.get(index).cloned()
    }
}
