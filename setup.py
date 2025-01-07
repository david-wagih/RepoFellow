from setuptools import setup, find_packages

setup(
    name="repo_fellow",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "graphviz",
        "langchain-community",
        "langchain-core",
        "langchain-openai",
        "langgraph",
        "langgraph-checkpoint-sqlite",
        "langgraph-cli",
        "langgraph-sdk",
        "langsmith",
        "matplotlib",
        "networkx",
        "notebook",
        "pygithub",
        "tavily-python",
        "trustcall",
        "wikipedia",
    ],
    python_requires=">=3.12",
) 