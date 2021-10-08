//! Helper functions to interpret string inputs.

use std::ffi::OsString;
use std::{fs::DirEntry, path::Path};

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

/// Attempts to interpret the given string as a directory and read its
/// contents. If that succeeds, then returns the last entry in the directory,
/// sorted alphabetically, which ends with `extension`. If no matching files
/// are found or if `dir` was not a directory, then returns `None`.
pub fn last_in_dir_ending_with(dir: &str, extension: &str) -> Option<String> {
    if let Ok(entries) = std::fs::read_dir(dir) {
        return if let Ok(mut entries) = entries.collect::<Result<Vec<DirEntry>, _>>() {
            entries.retain(|e| e.file_name().to_str().unwrap().ends_with(extension));
            entries.sort_by_key(DirEntry::file_name);
            entries
                .last()
                .map(DirEntry::file_name)
                .map(OsString::into_string)
                .map(Result::ok)
                .flatten()
        } else {
            None
        }
    }
    None
}
