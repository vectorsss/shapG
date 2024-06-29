from setuptools import setup, find_packages

setup(
    name="shapG",
    version="0.12.0",
    packages=find_packages(),
    install_requires=[
        "networkx",
        "matplotlib",
        "numpy",
        "tqdm",
        "pandas",
        "scipy",
        "numpy",
    ],
    extras_require={
        "dev": [
            "unittest",
        ],
    },
    author="Chi Zhao",
    author_email="dandanv5@hotmail.com",
    description="A library to compute Shapley values in graphs.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/vectorsss/shapG",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
