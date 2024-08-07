from setuptools import setup, find_packages

def get_requirements(filename):
    with open(filename, 'r') as file:
        requirements = file.read().splitlines()
    # Filter out '-e .'
    requirements = [req for req in requirements if req.strip() != '-e .']
    return requirements

setup(
    name='mlproject',
    version='0.1',
    author='Felipe Lacombe',
    author_email='felipedmlacombe@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)