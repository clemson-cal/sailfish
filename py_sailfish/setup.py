from distutils.core import setup, Extension

def main():
    srhd_1d = Extension(
        "srhd_1d",
        ["src/srhd_1d.c"],
        extra_compile_args=['-Xpreprocessor', '-fopenmp'],
        extra_link_args=['-lomp'],
    )

    setup(name="sailfish",
          version="0.4.0",
          description="GPU accelerated astrophysical gasdynamics code",
          author="Jonathan Zrake",
          author_email="jzrake@clemson.edu",
          ext_modules=[srhd_1d],
    )

if __name__ == "__main__":
    main()
