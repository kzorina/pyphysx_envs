import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyphysx_envs",
    version="0.3",
    author="Kateryna Zorina",
    author_email="zorina.kateryna.mail@gmail.com",
    description="Package contains a set of environments created with pyphysx library.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kzorina/pyphysx_envs",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['numpy', 'quaternion', 'rlpyt',
                      # 'pyphysx_utils@git+https://github.com/petrikvladimir/pyphysx_utils.git@master',
                      # 'pyphysx_render@git+https://github.com/petrikvladimir/pyphysx_render.git@master',
                      'pyphysx@git+https://github.com/petrikvladimir/pyphysx.git@master',
                      'rlpyt_utils@git+https://github.com/petrikvladimir/rlpyt_utils.git@master',
                      'torch',
                      'trimesh', 'pycollada==0.6']
)
