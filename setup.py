from setuptools import find_packages, setup
setup(
    name='pyctm',
    packages=find_packages(),
    version='0.0.8',
    description='Python Cognitive System Toolkit for Microservices',
    author='Eduardo de Moraes Froes',
    license='MIT',
    install_requires=['confluent-kafka', 'numpy', 'tk', 'matplotlib', 'torch', 'scikit-learn'],
    setup_requires=['pytest-runner', 'numpy', 'tk', 'matplotlib', 'torch', 'scikit-learn'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
    keywords=['python', 'cst']
)