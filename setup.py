from setuptools import setup, find_packages

setup(
    name="repo_fellow",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "graphviz>=0.20.3",
        "langchain-community>=0.3.14",
        "langchain-core>=0.3.29",
        "langchain-openai>=0.2.14",
        "langgraph>=0.2.61",
        "langgraph-checkpoint-sqlite>=2.0.1",
        "langgraph-cli>=0.1.65",
        "langgraph-sdk>=0.1.48",
        "langsmith>=0.2.10",
        "matplotlib>=3.10.0",
        "networkx>=3.4.2",
        "notebook>=7.3.2",
        "pygithub>=2.5.0",
        "tavily-python>=0.5.0",
        "trustcall>=0.0.26",
        "wikipedia>=1.4.0",
    ],
    python_requires=">=3.12",
) 