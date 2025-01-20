from setuptools import setup, find_packages

setup(
    name="kowpy",
    version="0.2.2",
    packages=find_packages(),
    install_requires=[
        "tree-sitter==0.21.3",
        "pandas>=2.0.0",
    ],
)
