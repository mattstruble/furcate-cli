from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read

with open('VERSION') as f:
    version = f.read

setup(
    name='furcate',
    version=version,
    description='A lightweight wrapper for automatically forking deep learning training sessions based on user configurations.',
    long_description=readme,
    author='Matt Struble',
    author_email='mattstruble@outlook.com',
    url='https://github.com/mattstruble/furcate',
    license=license,
    packages=find_packages(exclude=('tests', 'docs', 'example')),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ]
)