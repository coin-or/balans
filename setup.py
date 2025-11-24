import os

import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    required = f.read().splitlines()

with open(os.path.join('balans', '_version.py')) as f:
    exec(f.read())

# python setup.py bdist_wheel
setuptools.setup(
    name="balans",
    description="Balans: Bandit-based Adaptive Large Neighborhood Search",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=__version__,
    author="skadio",
    url="https://github.com/skadio/balns",
    packages=setuptools.find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests", "notebooks"]),
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.10+",
        "Operating System :: OS Independent",
    ],
    project_urls={
        "Documentation": "https://github.com/skadio/balans",
        "Source": "https://github.com/skadio/balans"
    }
)
