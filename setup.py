import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='wutools',
    version='0.0.2',
    author='Jason Yunger',
    author_email='jason.yunger@gmail.com',
    description='Testing installation of Package',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/jyunger/wutest',
    project_urls = {
        "Bug Tracker": "https://github.com/jyunger/wutest/issues"
    },
    license='MIT',
    packages=['wutools'],
    install_requires=['numpy', 'pandas', 'psutil', 'pyathena'],
)
