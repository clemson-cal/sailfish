from setuptools import setup

if __name__ == "__main__":
    setup(
        package_data=dict(sailfish=["solvers/*.c"]),
        entry_points={
            "console_scripts": ["sailfish=sailfish.driver:main"],
        },
    )
