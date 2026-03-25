"""
GDA-VisionAssist
Vision-Language System for Assisting Visually Impaired Individuals
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="GDA-VisionAssist",
    version="1.0.0",
    author="Khanh Chien Ngo",
    author_email="khanhchien6@gmail.com",
    description="Vision-Language System for Assisting Visually Impaired Individuals in Object Recognition and Description Query",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ChinhocIT/GDA-VisionAssist",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Intended Audience :: Science/Research",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.37.0",
        "numpy>=1.21.0",
        "opencv-python>=4.8.0",
        "Pillow>=9.0.0",
        "pyyaml>=6.0",
        "segment-anything-2",
        "edge-tts>=6.1.0",
        "pygame>=2.5.0",
        "SpeechRecognition>=3.10.0",
        "PyAudio>=0.2.13",
        "pynput>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
            "mypy>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "gda=app:main",
        ],
    },
)
