import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fourier-dnn", # Replace with your own username
    version="0.0.1",
    author="Samyak Sarnayak",
    author_email="samyaks210@gmail.com",
    description="Tensorflow 2.0 implementation of fourier feature mapping networks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Samyak2/fourier-dnn",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
