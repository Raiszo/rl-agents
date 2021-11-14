from setuptools import setup

setup(
    name='rl-agents',
    version='0.0.1',
    install_requires=[
        'tensorflow==2.5.1',
        'tensorflow-probability==0.12.1',
        'gym',
        'tensorboard==2.5',
        'tqdm==4.59.0'
    ]
)
