from setuptools import find_packages, setup

setup(
    name='tinyvector',
    version='0.1.0',
    author='Will DePue',
    author_email='will@depue.net',
    license='MIT',
    description='the tiny, least-dumb, speedy vector embedding database.',
    packages=find_packages(include=['core', 'core.*']),
    install_requires=[
        'numpy',
        'uuid',
        'pydantic',
        'psutil',
        'scikit-learn',
    ],
)
