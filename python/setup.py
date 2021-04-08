import os
import sys
from setuptools import setup, find_packages

packages = find_packages("src")

setup(name="logopy",
      version='1.0',
      packages=packages,
      license="MIT",
      author="Paloma Sodhi",
      package_dir={"": "src"}
      )