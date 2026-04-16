from setuptools import setup, find_packages

setup(
    name="tnd-intrinsic-value-model",
    version="1.0.0",
    description="Real-Time Intrinsic USD/TND Valuation Model — FIN 460",
    author="FIN 460 Project Team, Tunis Business School",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pandas>=2.0",
        "numpy>=1.24",
        "scipy>=1.10",
        "statsmodels>=0.14",
        "pykalman>=0.9.7",
        "yfinance>=0.2.36",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "scikit-learn>=1.3",
        "openpyxl>=3.1",
    ],
)
