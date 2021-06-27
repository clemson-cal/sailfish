use std::error;
use std::fmt;
use std::fmt::Display;

#[derive(Debug)]
pub enum Error {
    CompiledWithoutOpenMP,
    PrintUserInformation(String),
    CommandLineParse(String),
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
            Self::CommandLineParse(message) => {
                writeln!(fmt, "error: {}", message)
            }
            Self::IOError(error) => {
                error.fmt(fmt)
            }
        }
    }
}

impl error::Error for Error {}
