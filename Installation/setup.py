import setuptools
import codecs
import os.path

with open("readme.md", "r") as fh:
    long_description = fh.read()


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setuptools.setup(
    name="GiantFFTResample",  # Replace with your own username
    version=get_version("GiantFFTResample/__init__.py"),
    author="Peladeau",
    author_email="come.peladeau@telecom-paris.fr",
    description="Samplerate conversion using giant FFTs in numpy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/peladeaucome/giant_fft_resample",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.26.1",
        "torch>=2.0.0",
    ],
)