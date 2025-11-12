#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI フォント生成システム - セットアップスクリプト
"""

from setuptools import setup, find_packages
from pathlib import Path

# README読み込み
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")

# requirements.txt読み込み
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file, "r", encoding="utf-8") as f:
        requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

setup(
    name="fontgen-ai",
    version="0.1.0",
    description="AIを用いたフォント生成システム",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/fontgen-ai",
    license="MIT",
    packages=find_packages(exclude=["tests", "notebooks", "scripts"]),
    install_requires=requirements,
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "fontgen-ai=cli.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="font generation ai machine-learning pytorch typography",
)
