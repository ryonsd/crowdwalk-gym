from setuptools import setup, find_packages

setup(
    name="crowdwalk_gym",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy == 1.19.5",
        "pandas == 1.1.5",
        "matplotlib == 3.3.4",
        "gym == 0.17.3",
        "tensorboardX == 2.3",
        "pygmo == 2.16.1",
        "protobuf == 3.20.*",
    ],
    author="Ryo Nishida",
    description='An RL environment for a crowd route guidance using a crowd simulator.',
    url='https://github.com/ryonsd/crowdwalk-gym',
)
