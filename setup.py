import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="edcnlp",
    version="0.0.1",
    author="Edi Chen",
    author_email="yachen2016@gmail.com",
    description="A NLP research toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/edchengg/MyNLP",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)