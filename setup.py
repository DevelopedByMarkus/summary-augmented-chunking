from setuptools import setup, find_packages

setup(
    name="sac_rag",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    description="A retrieval-augmented generation (RAG) system using Summary Augmented Chunking (SAC).",
    python_requires=">=3.9",
)
