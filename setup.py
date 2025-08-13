from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="resume-segregator-pro",
    version="2.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-Powered Resume Segregator & Candidate Ranking System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/resume-segregator-pro",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Human Resources",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "black>=24.0.0",
            "flake8>=7.0.0",
            "pre-commit>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "resume-segregator=app:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

