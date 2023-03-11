# Copyright (C) 2023 Mohamed Azharudeen <azhar@neuralnexus.cloud>
# License: MIT, azhar@neuralnexus.cloud

from setuptools import find_packages, setup


def readme():
    with open("README.md", encoding="utf8") as f:
        README = f.read()
    return README


with open("requirements.txt") as f:
    required = f.read().splitlines()

with open("requirements-test.txt") as f:
    required_test = f.read().splitlines()
    
extras_require = {

    "test": required_test,
}

extras_require["full"] = (
    extras_require["test"]
)

setup(
    name="neuralnexus",
    version="0.1.0",
    description="neuralnexus - An open source, low-code machine learning library in Python.",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/azharlabs/neuralnexus",
    author="Mohamed Azharudeen",
    author_email="azhar@neuralnexus.cloud",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(include=["neuralnexus*"]),
    include_package_data=True,
    install_requires=required,
    extras_require=extras_require,
    tests_require=required_test,
    python_requires=">=3.8",
    test_suite='tests',
)