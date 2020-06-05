import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()


requirements = []


with open("requirements.txt", "r") as req:
    requirements = req.read().split()


setuptools.setup(
    name="tspymfe",
    version="0.0.3",
    author="Felipe Siqueira",
    author_email="felipe.siqueira@usp.br",
    description="Univariate time-series expansion for Pymfe package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FelSiq/ts-pymfe",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=requirements,
)
