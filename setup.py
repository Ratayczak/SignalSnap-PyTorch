from setuptools import setup, find_packages

setup(
    name='multichss',
    version='0.1.0',
    description='SignalSnap (PyTorch): Signal Analysis In Python Made Easy',
    author='Armin Ghorbanietemad',
    author_email='armin.ghorbanietemad@rub.de',
    license='BSD-3-Clause',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ArminGEtemad/SignalSnap2/tree/main/src/multichss',  
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)