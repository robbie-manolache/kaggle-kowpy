from setuptools import setup, find_packages

setup(
    name="kowpy",
    version="0.9.4",
    packages=find_packages(),
    install_requires=[
        "tree-sitter==0.21.3",
        "tree_sitter_languages",
        "pandas>=2.0.0",
        "transformers",
        "tokenizers",
    ],
)
