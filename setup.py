"""Setup file for Repo_Synopsis"""
from setuptools import setup, find_packages

setup(
    name="repo_synopsis",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
