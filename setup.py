from setuptools import find_packages, setup
setup(
    name='pyctm',
    packages=find_packages(include=['pyctm']),
    version='0.0.1',
    description='PyCTM library',
    author='Eduardo de Moraes Froes',
    license='MIT',
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)