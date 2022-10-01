import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='WFirmin',
    version='0.0.14',
    author='Will Firmin',
    author_email='will.firmin@hotmail.com',
    description='Personal Toolbox',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/WFirmin/Personal',
    license='MIT',
    packages=['WFirmin'],
    install_requires=["numpy","pandas","scipy","seaborn"],
)
