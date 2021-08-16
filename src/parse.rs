//! Helper functions to interpret string inputs.

use std::path::Path;

/// Takes a string, and splits it into two parts, separated by the first
/// instance of the given character. The first item in the pair is `Some`
/// unless the input string is empty. The second item in the pair is `None` if
/// `separator` is not found in the string.
pub fn split_pair(string: &str, separator: char) -> (Option<&str>, Option<&str>) {
    let mut a = string.splitn(2, separator);
    let n = a.next();
    let p = a.next();
    (n, p)
}

/// Takes a string, and splits it into two parts, separated by the first
/// instance of the given character. Each part is then parsed, and the whole
/// call fails if either one fails.
pub fn parse_pair<T: std::str::FromStr<Err = E>, E>(
    string: &str,
    separator: char,
) -> Result<(Option<T>, Option<T>), E> {
    let (sr1, sr2) = split_pair(&string, separator);
    let sr1 = sr1.map(|s| s.parse()).transpose()?;
    let sr2 = sr2.map(|s| s.parse()).transpose()?;
    Ok((sr1, sr2))
}

/// Returns the parent directory for an absolute path string, or `None` if no
/// parent directory exists. If the path is relative this function returns
/// `Some(".")`.
pub fn parent_dir(path: &str) -> Option<&str> {
    Path::new(path)
        .parent()
        .and_then(Path::to_str)
        .map(|s| if s.is_empty() { "." } else { s })
}
