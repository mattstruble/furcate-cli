from setuptools import find_packages, setup

with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

with open("VERSION") as f:
    version = f.read()

setup(
    name="furcate",
    version=version,
    description="A lightweight wrapper for automatically forking deep learning sessions to enable parallel model training across multiple GPUs.",
    long_description=readme,
    author="Matt Struble",
    author_email="mattstruble@outlook.com",
    url="https://github.com/mattstruble/furcate",
    license=license,
    packages=find_packages(exclude=("tests", "docs", "examples")),
    include_package_data=True,
    install_requires=["pandas", "matplotlib"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: GPU",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Build Tools",
        "Topic :: System :: Hardware :: Symmetric Multi-processing",
        "Topic :: Utilities",
    ],
)
