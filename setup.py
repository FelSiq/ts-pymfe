import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="tspymfe", # Replace with your own username
    version="0.0.1",
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
)
