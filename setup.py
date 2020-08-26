import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyphysx_envs",
    version="0.1",
    author="Kateryna Zorina",
    author_email="zorina.kateryna.mail@gmail.com",
    description="Package contains a set of environments created with pyphysx library.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kzorina/pyphysx_envs",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)