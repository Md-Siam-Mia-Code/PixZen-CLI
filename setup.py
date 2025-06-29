# setup.py
import re
import os
from setuptools import setup, find_packages


def get_version():
    version_file = os.path.join("pixzen", "__init__.py")
    with open(version_file, "r", encoding="utf-8") as f:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def get_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]


setup(
    name="pixzen",
    version=get_version(),
    author="Md. Siam-Mia",
    description="An AI-powered image and video enhancement tool using Real-ESRGAN and GFPGAN.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8, <4",
    install_requires=[req for req in get_requirements() if "onnxruntime" not in req],
    include_package_data=True,
    package_data={"pixzen": ["vendor/ffmpeg/*"]},
    entry_points={
        "console_scripts": [
            "pixzen = pixzen.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Topic :: Multimedia :: Graphics :: Graphics Conversion",
        "Environment :: Console",
    ],
)
