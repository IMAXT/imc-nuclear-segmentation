from setuptools import find_packages, setup

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'dask',
    'distributed',
    'voluptuous',
    'opencv-python',
    'numpy',
    'scipy',
    'scikit-image',
    'astropy',
    'pillow',
    'imaxt-image',
]

setup_requirements = ['pytest-runner', 'flake8']

test_requirements = ['coverage', 'pytest', 'pytest-cov', 'pytest-mock']

setup(
    author='Ali Dariush',
    author_email='adariush@ast.cam.ac.uk',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
    ],
    description='IMC pipeline.',
    entry_points={'owl.pipelines': 'imc = imc_pipeline'},
    install_requires=requirements,
    license='GNU General Public License v3',
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    data_files=[('conf', ['conf/imc_pipeline.yaml'])],
    keywords='imaxt, imc, owl',
    name='imc-pipeline',
    packages=find_packages(include=['imc_pipeline']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://gitlab.ast.cam.ac.uk/imaxt/imc_pipeline',
    version='0.3.0',
    zip_safe=False,
    dependency_links=['https://imaxt.ast.cam.ac.uk/pip/imaxt-image'],
)
