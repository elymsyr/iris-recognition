from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="iris_system",
    version="0.0.1",
    description="""Python Application for an Iris Recognition System using OpenCV. The project aims to create a database system and iris analyzer for fast fast and accurate iris recognition. See for more, https://github.com/elymsyr/iris-recognition.""",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Orhun Eren Yalçınkaya",
    packages=find_packages(include=["iris_system"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=required,
    license="MIT",
    url="https://github.com/elymsyr/iris-recognition",
)