use std::error;
use std::fmt;
use std::fmt::Display;

#[derive(Debug)]
pub enum Error {
    CompiledWithoutOpenMP,
    CommandLineInterrupt(String),
}

impl Display for Error {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        match self {
            Self::CompiledWithoutOpenMP => {
                write!(fmt, "compiled without OpenMP support")
            }
            Self::CommandLineInterrupt(message) => {
                write!(fmt, "{}", message)
            }
        }
    }
}

impl error::Error for Error {}
