from setuptools import setup, find_packages  # type: ignore


setup(
    name='atbdump',
    version='0.1.0',
    description='atb dump tool',
    url='https://gitee.com/ascend/ait/ait/components/atbdump',
    packages=find_packages(),
    license='Apache-2.0',
    keywords='atbdump',
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
        'atbdump_sub_task': ['atbdump=atbdump.__main__:get_cmd_instance'],
    },
)