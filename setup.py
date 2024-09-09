from setuptools import setup, find_packages


# Read requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
   name='simple-cats',
   version='0.1.2',
   package_dir={'simple_cats': 'src'},
   packages=find_packages(where='src'),
   description='Python package for CATS paper',
   long_description=open('README.md').read(),
   long_description_content_type='text/markdown',
   author='Jeyong Lee, Donghyun Lee, ...',
   author_email='je-yong.lee@worc.ox.ac.uk',
   url='https://github.com/ScalingIntelligence/CATS',
   install_requires=['transformers', 'torch', 'triton', ],
   classifiers=[
       'Programming Language :: Python :: 3',
       'License :: OSI Approved :: MIT License',
       'Operating System :: OS Independent',
   ],
   python_requires='>=3.6',
)