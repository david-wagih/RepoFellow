from setuptools import setup, find_packages

setup(
    name="code-assistant",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "typer",
        "rich",
        "requests",
        "pathlib",
    ],
    entry_points={
        "console_scripts": [
            "code-assistant=main:app",
        ],
    },
) 