import runpy
import logging

from setuptools import _install_setup_requires, setup, find_packages

PACKAGE_NAME = "clifter-slam"
VERSION = runpy.run_path("clifter_slam/version.py")["__version__"]
DESCRIPTION = "ClifterSlam: Dense Slam mmets automatic differentation"
URL = "< url.to.go.in.here>"
AUTHOR = "ArfySlowy"
LICENSE = "MIT"
DOWNLOAD_URL = ""
LONG_DESCRIPTION = """
Clifterslam is dense slam technology
"""
CLASSIFIERS = [
    "Intended Audience :: Developers",
    "Intended Audience :: Education",
    "Intended Audience :: Science/Research",
    "Operating System :: POSIX :: Linux",
    "Programming Language :: Python :: 3 :: Only",
    "License :: OSI Approved :: MIT",
    "Topic :: Software Development :: Libraries",
]

logger = logging.getLogger()
logging.basicConfig(format="%(levelname)s - %(message)s")


def get_requirements():
    packages = None
    with open("requirements.txt") as file:
        packages = file.read().splitlines()
    return packages


if __name__ == "__main__":
    setup(
        name=PACKAGE_NAME,
        version=VERSION,
        author=AUTHOR,
        description=DESCRIPTION,
        url=URL,
        long_description=LONG_DESCRIPTION,
        license=LICENSE,
        python_requires=">3.6",
        packages=find_packages(exclude=("test", "example")),
        install_requires=get_requirements(),
        extras_require={
            "all": ["matplot", "tqdm"],
            "dev": [
                "black",
                "flake8",
                "nbsphinx",
                "pytest>=4.6",
                "pytest-cov>=2.7",
                "sphinx==2.2.0 ",
            ],
        },
        zip_safe=True,
        include_dirs=[],
        classifiers=CLASSIFIERS,
    )
