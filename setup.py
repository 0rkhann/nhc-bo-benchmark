from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="nhc-bo-benchmark",
    version="0.1.0",
    author="Orkhan Abdullayev",
    description="Bayesian Optimization for Molecular Property Optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/0rkhann/nhc-bo-benchmark",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[req.split(">=")[0] for req in requirements if not req.startswith("#")],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "ml": [l.strip() for l in open("requirements-ml.txt") if "==" in l],
        "optional": [
            "xgboost>=1.6.0",
            "pyopls>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "bo-optimize=src.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml"],
    },
)
