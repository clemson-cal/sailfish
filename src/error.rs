use std::error;
use std::fmt;
use std::fmt::Display;
use std::num::ParseFloatError;

#[derive(Debug)]
pub enum Error {
    CompiledWithoutOpenMP,
    CompiledWithoutGpu,
    PrintUserInformation(String),
    Cmdline(String),
    UnknownEnumVariant { enum_type: String, variant: String },
    InvalidSetup(String),
    InvalidCheckpoint(String),
    InvalidDevice(i32),
    ParseFloatError(ParseFloatError),
    IOError(std::io::Error),
}

impl Display for Error {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        match self {
            Self::PrintUserInformation(message) => {
                write!(fmt, "{}", message)
            }
            Self::CompiledWithoutOpenMP => {
                writeln!(fmt, "error: built without OpenMP support")
            }
            Self::CompiledWithoutGpu => {
                writeln!(fmt, "error: built without GPU support")
            }
            Self::Cmdline(message) => {
                writeln!(fmt, "error: {}", message)
            }
            Self::UnknownEnumVariant { enum_type, variant } => {
                writeln!(fmt, "unknown mode '{}' for {}", variant, enum_type)
            }
            Self::InvalidSetup(info) => {
                writeln!(fmt, "invalid setup: {}", info)
            }
            Self::InvalidCheckpoint(info) => {
                writeln!(fmt, "invalid checkpoint: {}", info)
            }
            Self::InvalidDevice(id) => {
                writeln!(fmt, "invalid device id: {}", id)
            }
            Self::ParseFloatError(error) => {
                writeln!(fmt, "{}", error)
            }
            Self::IOError(error) => {
                writeln!(fmt, "{}", error)
            }
        }
    }
}

impl error::Error for Error {}
