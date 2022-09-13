import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='Toolbox',
    version='0.0.4',
    author='Will Firmin',
    author_email='will.firmin@hotmail.com',
    description='Testing installation of Package',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/WFirmin/Personal',
    license='MIT',
    packages=['Toolbox'],
    install_requires=[],
)
