import sys, os
from setuptools import setup, find_packages
from pathlib import Path

# The long_description is now specified in pyproject.toml via the 'readme' field.
# The 'packages', 'package_data', 'install_requires', 'scripts',
# and other metadata are also handled in pyproject.toml.

setup(
    # setuptools will read most of the metadata from pyproject.toml
    # You might still need to explicitly define packages if find_packages()
    # doesn't correctly identify all of them based on your project structure.
    packages=find_packages(), # This will automatically find all packages
    zip_safe=False,
    # Any custom build logic or specific configurations not easily expressed in pyproject.toml
    # can still reside here.
)