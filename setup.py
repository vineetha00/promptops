from setuptools import setup, find_packages
from pathlib import Path

here = Path(__file__).parent
long_description = (here / "README.md").read_text() if (here / "README.md").exists() else ""

setup(
    name="promptops",
    version="0.1.1",
    description="Git-native version control for LLM prompts: semantic diff, Thompson Sampling A/B, LLM-as-judge eval, MCP server.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Vineetha Joy",
    author_email="vineetha@usc.edu",
    url="https://github.com/vineetha00/promptops",
    license="MIT",
    packages=find_packages(exclude=("tests", "tests.*")),
    python_requires=">=3.8",
    install_requires=[
        "click>=8.0",
        "sentence-transformers>=2.0",
        "scipy>=1.9",
        "rich>=13.0",
        "anthropic>=0.20",
        "mcp>=1.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "twine>=4.0", "build>=1.0"],
    },
    entry_points={
        "console_scripts": ["promptops=promptops.cli:main"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Version Control",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="llm prompt-engineering version-control mcp ab-testing",
)
