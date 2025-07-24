from setuptools import setup, find_packages

setup(
    name="promptpilot",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # List your project's dependencies here
        "pandas",
        "python-dotenv",
        "google-generativeai",
    ],
    python_requires=">=3.8",
)
