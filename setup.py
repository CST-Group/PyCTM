from setuptools import find_packages, setup
setup(
    name='pyctm',
    packages=find_packages(),
    version='0.0.7',
    description='Python Cognitive System Toolkit for Microservices',
    author='Eduardo de Moraes Froes',
    license='MIT',
    install_requires=['confluent-kafka'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
    keywords=['python', 'cst']
)
