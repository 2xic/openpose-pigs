import setuptools

packages = setuptools.find_packages()
setuptools.setup(
    name="openpose",
    version="0.0.2",
    author_email="author@example.com",
    description="openpose package",
    packages=packages,
    python_requires='>=3.6',
)
