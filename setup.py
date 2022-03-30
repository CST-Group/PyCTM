from setuptools import find_packages, setup
setup(
    name='pyctm',
    packages=find_packages(include=['pyctm']),
    version='0.0.1',
    description='Python Cognitive Toolkit for Microservices',
    author='Eduardo de Moraes Froes',
    license='MIT',
    install_requires=[],
    setup_requires=['pytest-runner', 'confluent-kafka'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)
