from setuptools import setup, find_packages

# Read the contents of your requirements.txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='fab',
    version='0.1.0',
    author='Fabio Catania',
    author_email='fabiocat@mit.edu',
    description='fab - short for Fabio\'s Audio Box - is a python package for processing and analyzing voice and speech data. Whether you\'re a seasoned researcher or a curious enthusiast, fab enables you to perform experiments with a focus on functional audio processing.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    package_data={
        "fab": ["tasks/*/schemas/*.json", 
                "pipelines/*/schemas/*.json"], # If any submodule contains json-schema files, include them:
    },
    license='LICENSE',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],
    install_requires=requirements,  # This is the list read from the requirements.txt file
)
