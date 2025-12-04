from setuptools import setup, find_packages

setup(
    name="llm-rl-code-golf",
    version="0.1.0",
    description="LLM-based Reinforcement Learning for Google Code Golf 2025",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        line.strip()
        for line in open("requirements.txt").readlines()
        if line.strip() and not line.startswith("#")
    ],
)
