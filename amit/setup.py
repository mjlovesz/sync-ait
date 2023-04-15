from setuptools import setup, find_packages  # type: ignore

with open('requirements.txt', encoding='utf-8') as f:
    required = f.read().splitlines()

setup(
    name='amit',
    version='0.0.1',
    install_requires=required,
    description='AMIT, Ascend MindStudio Inference Tools',
    long_description_content_type='text/markdown',
    url='https://gitee.com/ascend/amit',
    packages=find_packages(),
    package_data={'': ['LICENSE', 'model.cfg']},
    license='Apache-2.0',
    keywords='amit',
    python_requires='>=3.7',
    entry_points={
        'console_scripts': ['amit=.__main__:cli'],
    },
)
