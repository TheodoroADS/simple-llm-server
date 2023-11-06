from setuptools import find_packages, setup

setup(
    name='llm_client',
    packages=find_packages(include=["llm_client"]),
    version='0.1.0',
    description='The client for the local llm server',
    author='Theodoro Martin Ahualli',
)