from setuptools import setup

setup(
    name='rl-agents',
    version='0.0.1',
    install_requires=[
        'tensorflow',
        'tensorlfow_probability',
        'gym[box2d]',
        'tensorboard',
    ]
)
