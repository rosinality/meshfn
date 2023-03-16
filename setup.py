from setuptools import setup, find_packages

setup(
    name="meshfn",
    version="0.1.0",
    description="Configuration and distributed training library for PyTorch",
    url="https://github.com/rosinality/meshfn",
    author="Kim Seonghyeon",
    author_email="kim.seonghyeon@outlook.com",
    license="MIT",
    install_requires=[
        "pydantic>=1.8",
        "rich",
    ],
    packages=find_packages(),
)
