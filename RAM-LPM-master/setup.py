import setuptools

setuptools.setup(
    name="tonbo",
    version="1.0.0",
    python_requires=">3.6",
    author="Koji Ono",
    author_email="koji.ono@exwzd.com",
    description="code for Kiritani and Ono, 2020",
    packages=setuptools.find_packages(),
    install_requires=[],
    setup_requires=["numpy", "pytest-runner"],
    tests_require=["pytest-cov", "pytest-html", "pytest"],
    classifiers=["Programming Language :: Python :: 3.6",],
)
