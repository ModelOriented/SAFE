import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="safe-transformer",
    version="0.0.4",
    author="Aleksandra Gacek, Piotr Luboń",
    author_email="lubonp@student.mini.pw.edu.pl, gaceka@student.mini.pw.edu.pl",
    description="Build explainable ML models using surrogate models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/olagacek/SAFE",
    packages=setuptools.find_packages(),
    install_requires=[
          'numpy',
          'ruptures',
          'sklearn',
          'pandas',
          'scipy',
          'kneed'
      ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
