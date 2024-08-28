from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="iris_recognizer_system",
    version="0.0.1",
    description="""Iris Recognizer System allows you to analyze iris images and compare them. Also it brings a database control.""",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Orhun Eren Yalçınkaya",
    packages=find_packages(include=["iris_recognizer_system"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=required,
    license="MIT",
    url="https://github.com/elymsyr/iris-recognition",
)