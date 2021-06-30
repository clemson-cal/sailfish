use std::error;
use std::fmt;
use std::fmt::Display;

#[derive(Debug)]
pub enum Error {
    CompiledWithoutOpenMP,
    PrintUserInformation(String),
    Cmdline(String),
    InvalidSetup(String),
    InvalidCheckpoint(String),
    IOError(std::io::Error),
}

impl Display for Error {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        match self {
            Self::PrintUserInformation(message) => {
                write!(fmt, "{}", message)
            }
            Self::CompiledWithoutOpenMP => {
                writeln!(fmt, "error: compiled without OpenMP support")
            }
            Self::Cmdline(message) => {
                writeln!(fmt, "error: {}", message)
            }
            Self::InvalidSetup(info) => {
                writeln!(fmt, "invalid setup: {}", info)
            }
            Self::InvalidCheckpoint(info) => {
                writeln!(fmt, "invalid checkpoint: {}", info)
            }
            Self::IOError(error) => {
                writeln!(fmt, "{}", error)
            }
        }
    }
}

impl error::Error for Error {}
