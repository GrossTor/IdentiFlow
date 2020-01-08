import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="identiflow",
    version="0.0.1",
    author="Torsten Gross",
    author_email="gross.torsten1@gmail.com",
    description="Identifiability and experimental design in reverse-engineering of directed networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GrossTor/IdentiFlow",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
       'networkx>=2.4',
       'numpy>=1.17',
       'matplotlib>=3.1',
    ],
)
