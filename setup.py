import setuptools

with open('README.md') as f:
  long_description = f.read()
description = 'Solve the lake monster problem with RL.'

install_requires = [
    'tqdm',
    'tensorboard',
    'tf_agents',
    'pandas',
    'Pillow',
    'imageio_ffmpeg'
]

setuptools.setup(
    name='lake-monster',
    author='Zeb Engberg',
    author_email='zebengberg@gmail.com',
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/zebengberg/lake-monster',
    packages=setuptools.find_namespace_packages(exclude=['tests*']),
    python_requires='>=3.8.0',
    install_requires=install_requires,
    version='0.1.0',
    classifiers=['License :: OSI Approved :: MIT License',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.8',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence',
                 'Topic :: Games/Entertainment :: Simulation'],
    license='MIT')
