from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="scube-llm-utils",
    version="0.1.0",
    author="Tan Kuan Pern",
    author_email="kptan86@gmail.com",
    description="A collection of utility functions for LLM development",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kuanpern/scube-llm-utils",
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        "langchain>=1.0",
        "jinja2>=3.0.0",
        "PyYAML>=6.0",
        "tenacity>=9.0.0",
        "transformers>=4.55",
        "markdown>=3.10"
    ],
)