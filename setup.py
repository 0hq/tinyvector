from setuptools import find_packages, setup

setup(
    name='tinyvector',
    version='0.1.1',
    author='Will DePue',
    author_email='will@depue.net',
    license='MIT',
    description='the tiny, least-dumb, speedy vector embedding database.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/0hq/tinyvector',
    packages=find_packages(include=['core', 'core.*']),
    install_requires=[
        'numpy',
        'pydantic',
        'psutil',
        'scikit-learn',
    ],
)
