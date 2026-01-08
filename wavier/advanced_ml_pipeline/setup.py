"""Setup configuration for the advanced ML pipeline."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = [
        line.strip()
        for line in requirements_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="advanced-ml-pipeline",
    version="1.0.0",
    author="ML Engineer",
    author_email="ml@example.com",
    description="Production-grade machine learning pipeline with PyTorch and MLOps",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/advanced-ml-pipeline",
    packages=find_packages(exclude=["tests", "tests.*", "notebooks"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.0",
            "hypothesis>=6.92.0",
            "black>=23.11.0",
            "mypy>=1.7.0",
            "ruff>=0.1.6",
            "pre-commit>=3.5.0",
            "jupyter>=1.0.0",
            "jupyterlab>=4.0.0",
        ],
        "cuda": [
            "torch>=2.1.0+cu121",
        ],
    },
    entry_points={
        "console_scripts": [
            "ml-train=src.main:main",
            "ml-serve=src.inference.api:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
