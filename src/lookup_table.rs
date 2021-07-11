use std::num::ParseFloatError;
use std::fs::read_to_string;

static OFF_BOUNDS_RELATIVE_TOL: f64 = 1e-12;

/// An error type for failed ASCII table lookups
#[derive(thiserror::Error, Debug)]
pub enum Error {

    #[error(transparent)]
    ParseFloatError(#[from] ParseFloatError),

    #[error(transparent)]
    IoError(#[from] std::io::Error),

    #[error("the left-most column of the table must increase monotonically")]
    UnorderedTable,

    #[error("the table must have at least two rows")]
    TableTooSmall,
}

/// Encapsulates the logic to read and sample a tabulated function of a single
/// variable. The tabulated function range is a vector of `NUM_COLS - 1`
/// floats. Sampling the function returns a statically allocated array of
/// length `NUM_COLS`, i.e. the 0th element is the sample coordinate itself.
#[derive(Clone)]
pub struct LookupTable<const NUM_COLS: usize> {
    rows: Vec<[f64; NUM_COLS]>,
}

impl<const NUM_COLS: usize> LookupTable<NUM_COLS> {
    /// Returns a lookup table from a `Vec` of rows. A `TableTooSmall` error
    /// is returned if there are fewer than 2 rows, and `UnorderedTable` error
    /// is returned if the left-most column does not increasing monotonically.
    pub fn from_rows(rows: Vec<[f64; NUM_COLS]>) -> Result<Self, Error> {
        if rows.len() < 2 {
            return Err(Error::TableTooSmall)
        }
        let mut x_prev = rows.first().unwrap()[0];
        for row in &rows[1..] {
            let x = row[0];
            if x <= x_prev {
                return Err(Error::UnorderedTable)
            }
            x_prev = x;
        }
        Ok(Self { rows })
    }

    /// Creates a `LookupTable` by reading a string of ASCII data. The string
    /// must be the contents of a .dat-like file, with whitespace-separated
    /// floats. The input string _should_ have `NUM_COLS` floats per row, but
    /// newlines are not enforced; whitespace separated floats are simply
    /// consumed in groups of `NUM_COLS`. `std::num::ParseFloatError` is
    /// retured if any of the entries in the table failed to parse.
    pub fn from_ascii_table(contents: &str) -> Result<Self, Error> {
        let values: Result<Vec<_>, _> = contents.split_whitespace().map(|x| x.parse()).collect();
        let rows = values?
            .chunks(NUM_COLS)
            .map(|chunk| {
                let mut rows = [0.0; NUM_COLS];

                for i in 0..NUM_COLS {
                    rows[i] = chunk[i]
                }
                rows
            })
            .collect();
        Self::from_rows(rows)
    }

    /// Convenience method to load the contents of an ASCII file and pass the
    /// resulting string to `LookupTable::from_ascii_table`. `std::io::Error`
    /// is returned if the file could not be opened.
    pub fn from_ascii_file(filename: &str) -> Result<Self, Error> {
        Self::from_ascii_table(&read_to_string(filename)?)
    }

    /// Returns a fixed-length array of data at the given independent variable
    /// value `x`. The result is interpolated linearly between the two nearest
    /// tabulated points. This function panics if `x` is out of range (not
    /// between the lowest and highest value of the left-most table column.
    pub fn sample(&self, x: f64) -> [f64; NUM_COLS] {
        let mut result = [0.0; NUM_COLS];
        let (i0, i1) = self.indexes_straddling(x);
        let v = &self.rows;

        for i in 0..NUM_COLS {
            let x0 = v[i0][0];
            let y0 = v[i0][i];
            let x1 = v[i1][0];
            let y1 = v[i1][i];
            result[i] = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
        }
        result
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn rows(&self) -> &[[f64; NUM_COLS]] {
        &self.rows
    }

    fn indexes_straddling(&self, mut x: f64) -> (usize, usize) {
        let xmin = self.rows.first().unwrap()[0];
        let xmax = self.rows.last().unwrap()[0];

        if x <= xmin {
            let norm = (x.abs() + xmin.abs()).max(1.0);
            if (xmin - x) / norm < OFF_BOUNDS_RELATIVE_TOL {
                x = xmin * (1.0 + OFF_BOUNDS_RELATIVE_TOL);
            }
            else {
                panic! {
                    "attempt to sample table at or below smallest tabulated point ({} <= {})",
                    x,
                    xmin
                }
            }
        }
        if x > xmax {
            let norm = (x.abs() + xmax.abs()).max(1.0);
            if (x - xmax) / norm < OFF_BOUNDS_RELATIVE_TOL {
                x = xmax * (1.0 - OFF_BOUNDS_RELATIVE_TOL);
            } else {
                panic! {
                    "attempt to sample table above the largest tabulated point ({} > {})",
                    x,
                    xmax
                }
            }
        }

        let index = match self
            .rows
            .binary_search_by(|row| Self::compare_f64(row[0], x))
        {
            Ok(index) => index,
            Err(index) => index,
        };
        (index - 1, index)
    }

    fn compare_f64(a: f64, b: f64) -> std::cmp::Ordering {
        if a < b {
            std::cmp::Ordering::Less
        } else if a > b {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic]
    fn lookup_table_panics_if_sampled_at_lower_bound() {
        let table = LookupTable::from_rows(vec![[0.0, 0.1], [1.0, 0.2], [2.0, 0.3]]).unwrap();
        table.indexes_straddling(0.0);
    }

    #[test]
    fn lookup_table_does_not_panic_if_sampled_at_upper_bound() {
        let table = LookupTable::from_rows(vec![[0.0, 0.1], [1.0, 0.2], [2.0, 0.3]]).unwrap();
        table.indexes_straddling(2.0);
    }

    #[test]
    fn lookup_table_panics_if_input_is_not_increasing() {
        assert!(LookupTable::from_rows(vec![[1.0, 0.1], [1.0, 0.2], [2.0, 0.3]]).is_err())
    }

    #[test]
    fn lookup_table_panics_if_input_is_not_ordered() {
        assert!(LookupTable::from_rows(vec![[1.0, 0.1], [0.0, 0.2], [2.0, 0.3]]).is_err())
    }

    #[test]
    fn lookup_table_gives_the_right_indexes_straddling() {
        let table = LookupTable::from_rows(vec![[0.0, 0.1], [1.0, 0.2], [2.0, 0.3]]).unwrap();
        assert_eq!(table.indexes_straddling(0.5), (0, 1));
        assert_eq!(table.indexes_straddling(1.0), (0, 1));
        assert_eq!(table.indexes_straddling(1.5), (1, 2));
    }

    #[test]
    fn lookup_table_can_be_sampled_at_tabulated_points() {
        let table = LookupTable::from_rows(vec![[0.0, 0.1], [1.0, 0.2], [2.0, 0.3]]).unwrap();
        assert!(f64::abs(table.sample(0.5)[1] - 0.15) < 1e-10);
        assert!(f64::abs(table.sample(1.0)[1] - 0.20) < 1e-10);
        assert!(f64::abs(table.sample(1.5)[1] - 0.25) < 1e-10);
    }
}
