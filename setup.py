from setuptools import setup, find_packages

setup(
    name="promptpilot",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit==1.46.0",
        "pandas==2.2.1",
        "google-generativeai==0.3.2",
        "python-dotenv==1.0.1",
        "protobuf==4.25.8",
        "typing-extensions==4.14.1",
        "pydantic==2.11.7",
    ],
    python_requires=">=3.8",
)
