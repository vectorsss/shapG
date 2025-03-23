from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="shapG",
    version="0.13.3",
    packages=find_packages(),
    install_requires=[
        "networkx",
        "matplotlib",
        "numpy",
        "tqdm",
        "pandas",
        "scipy",
        "numpy",
        "tabulate"
    ],
    extras_require={
        "dev": [
            "unittest",
        ],
    },
    author="Chi Zhao",
    author_email="dandanv5@hotmail.com",
    description="A library to compute Shapley values in graphs.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/vectorsss/shapG",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires='>=3.9',
)
