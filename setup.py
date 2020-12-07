import fnmatch
import sys

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py as build_py_orig

with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

with open("VERSION") as f:
    version = f.read()

description_template = "A lightweight wrapper for automatically forking {} sessions to enable parallel model training across multiple GPUs."
install_requires = ["pandas", "matplotlib"]
if "tf" in sys.argv:
    name = "furcate-tf"
    description = description_template.format("TensorFlow")
    install_requires.append("tensorflow >= 2.0")
    excludes = ("tests", "docs", "examples")
    sys.argv.remove("tf")
else:
    name = "furcate"
    description = description_template.format("deep learning")
    excludes = ("tests", "docs", "examples", "*.furcate-tf")


class build_py(build_py_orig):
    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        return [
            (
                pkg,
                mod,
                file,
            )
            for (
                pkg,
                mod,
                file,
            ) in modules
            if not any(
                fnmatch.fnmatchcase(pkg + "." + mod, pat=pattern)
                for pattern in excludes
            )
        ]


setup(
    name=name,
    version=version,
    description=description,
    long_description=readme,
    author="Matt Struble",
    author_email="mattstruble@outlook.com",
    url="https://github.com/mattstruble/furcate",
    license=license,
    packages=find_packages(exclude=excludes),
    cmdclass={"build_py": build_py},
    include_package_data=True,
    install_requires=install_requires,
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
