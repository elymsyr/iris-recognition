from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="iris_system",
    version="0.1.0",
    description="""Developing a robust tool for iris image analysis and recognition, including planned features such as advanced extraction and comparison of iris data, performance optimization using Random Forest Classifiers over keypoints, improvements for challenging conditions and database control. See for more, https://github.com/elymsyr/iris-recognition.""",
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
