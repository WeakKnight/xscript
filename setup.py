"""
XScript Package Setup

Install with: pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text(encoding='utf-8')

setup(
    name="xscript",
    version="0.1.0",
    author="XScript Team",
    description="GPU-accelerated scripting language for game development",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/xscript",
    packages=find_packages(),
    package_data={
        'xscript': [
            'runtime/*.slang',
            'stdlib/*.xs',
        ],
    },
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
    ],
    extras_require={
        "gpu": [
            "slangpy>=0.1.0",
        ],
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "mypy>=0.900",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Games/Entertainment",
        "Topic :: Software Development :: Compilers",
        "Topic :: Software Development :: Interpreters",
    ],
    keywords="scripting, gpu, game-development, slang, lua-like",
    entry_points={
        "console_scripts": [
            "xscript=xscript.cli:main",
        ],
    },
)

