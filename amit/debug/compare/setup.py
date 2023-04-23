from setuptools import setup, find_packages  # type: ignore

with open('requirements.txt', encoding='utf-8') as f:
    required = f.read().splitlines()

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='msquikcmp',
    version='0.0.1',
    description='This tool enables one-click network-wide accuracy analysis of TensorFlow and ONNX models.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://gitee.com/ascend/auto-optimizer',
    packages=find_packages(),
    package_data={'': ['LICENSE']},
    license='Apache-2.0',
    keywords='msquikcmp',
    install_requires=required,
    classifiers=[
        'Development Status :: Alpha',
        'Intended Audience :: Developers',
        'License :: Apache-2.0 Software License',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development'
    ],
    python_requires='>=3.7',
    entry_points={
        'console_scripts': ['msquikcmp=compare.__main__:main'],
        'debug_sub_task': ['compare=compare.__main__:main'],
    },
)
