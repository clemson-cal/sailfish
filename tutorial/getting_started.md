# Installation

## System dependencies

Sailfish requires Python 3, version 3.9.0 or later. You can check your
system's Python version by typing

```bash
python3 --version
```

On MacOS, I recommend using [Homebrew](https://brew.sh) to install the latest Python:

```bash
brew install python3
```

__Do not use Anaconda if you can help it__ (not that on Palmetto we do currently use Anaconda, because it is the only choice for Python >= 3.9). Use `pip3` to install Python module dependencies. The following command should install all the Python modules you'll need:

```bash
pip3 install numpy matplotlib cffi pydantic rich textual cpuinfo
```

If you are on Windows or Linux and have a CUDA or ROCm enabled GPU, you will also need [`cupy`](https://cupy.dev). For example, on an NVIDIA system with CUDA driver version 11.4 you would do
```bash
pip3 install cupy-cuda114
```
There is more information about `cupy` installation [here](https://docs.cupy.dev/en/stable/install.html).

## Clone the Sailfish repository

Sailfish is hosted on GitHub and requires `git` (Linux and MacOS systems will have that already). To get a local copy of the Sailfish code base, create a directory on your machine (I use `/Users/jzrake/Work/Codes` for science codes), then change directory there, and then clone the repository:

```
mkdir /Users/<your-username>/Work/Codes
cd /Users/<your-username>/Work/Codes
git clone git@github.com:clemson-cal/sailfish.git --branch=v0.6-beta
cd sailfish
```

Now you are in the Sailfish project directory where you can run the code, and also browse or modify the source code. The `sailfish` executable is located at `bin/sailfish`. Try running it with no arguments to see a help message:

```bash
/bin/sailfish
```
If that works, then you are ready to start doing science!
