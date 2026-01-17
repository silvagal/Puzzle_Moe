from setuptools import find_packages, setup

setup(
    name="puzzle-moe",
    version="0.1.0",
    description="Self-Supervised Symbolic Mixture of Experts for ECG Analysis",
    author="Research Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "scipy",
        "torch>=2.0",
        "pyyaml",
        "tqdm",
        "pandas",
        "wfdb",
        "scikit-learn",
    ],
    python_requires=">=3.9",
)
