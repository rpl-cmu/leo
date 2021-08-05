from setuptools import setup, find_packages

install_requires = [line.rstrip() for line in open("requirements/requirements.txt", "r")]

setup(
    name="leopy",
    version="0.0.1",
    description="Learning energy based models in graph optimization",
    url="",
    author="Paloma Sodhi",
    author_email="psodhi@cs.cmu.edu",
    license="LICENSE",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=install_requires,
    python_requires=">=3.6",
)