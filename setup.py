from setuptools import setup, find_packages

setup(
    name='buildabrainwave',
    version='0.0.0',
    description="Code for a the Nerd Nite talk, Feb 2018.",
    url='',
    author='Erik Peterson',
    author_email='erik.exists@gmail.com',
    license='MIT',
    classifiers=[
        'Intended Audience :: Science/Research', 'License :: OSI Approved',
        'Programming Language :: Python', 'Topic :: Scientific/Engineering',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX', 'Operating System :: Unix',
        'Operating System :: MacOS', 'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7'
    ],
    packages=find_packages(),
    install_requires=['numpy>=1.8.0', 'scipy>=0.15.1'], )
