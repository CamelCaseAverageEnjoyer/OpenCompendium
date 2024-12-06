from setuptools import setup, find_packages

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / 'README.md').read_text()

setup(name='near_earth_coordinates',
      version='1.0.0',
      python_requires='>=3',
      author='Orlov Vladislav',
      author_email='orlov.vv@phystech.edu',
      packages=find_packages(),
      install_requires=['numpy'],
      py_modules=['near_earth_coordinates', 'testing'],
      long_description=long_description,
      long_description_content_type='text/markdown',
      
     )



