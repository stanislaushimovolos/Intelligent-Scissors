import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Intelligent-Scissors",
    version="0.0.1",
    author="Stanislav Shimovolos",
    author_email='shimovolos.sa@phystech.edu',
    description="Intelligent Scissors tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT',
    url="https://github.com/stanislaushimovolos/Intelligent-Scissors",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'Dijkstar',
    ],
    python_requires='>=3.6',
)
