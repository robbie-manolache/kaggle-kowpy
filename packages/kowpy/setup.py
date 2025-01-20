from setuptools import setup, find_packages

setup(
    name="kowpy",
    version="0.2.1",
    packages=find_packages(),
    install_requires=[
        "tree-sitter",
        "pandas>=2.0.0",
    ],
)
