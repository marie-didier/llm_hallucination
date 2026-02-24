from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="llm-hallucination-detection",
    version="0.1.0",
    description="LLM Hallucination Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/marie-didier/llm_hallucination.git",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
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
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "gpu": [
            "torch>=2.0.0",
            "cudatoolkit>=11.7",
        ],
        "cpu": [
            "torch>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "run-nli=src.compute_nli:main",
            "run-kle=src.compute_kle:main",
            "evaluate-hallucination=src.evaluate:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)