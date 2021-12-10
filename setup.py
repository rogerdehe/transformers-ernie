# -*- coding: utf-8 -*-

from setuptools import find_packages, setup


# when modifying the following list, make sure to update src/transformers/dependency_versions_check.py
install_requires = [
    "transformers>4.0",
]

setup(
    name="transformers-ernie",
    version="0.1.0",
    author="roger",
    author_email="rogerdehe@gmail.com",
    description="Ernie model implemented by PyTorch",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="ernie pytorch transformers",
    license="Apache",
    url="",
    package_dir={"": "src"},
    packages=find_packages("src"),
    zip_safe=False,
    python_requires=">=3.6.0",
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
