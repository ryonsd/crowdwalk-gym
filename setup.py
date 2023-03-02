from setuptools import setup, find_packages

setup(
    name="crowdwalk_gym",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "gym",
        "tensorboardX >= 2.3",
        "pygmo >= 2.16",
        "protobuf",
    ],
    author="Ryo Nishida",
    description='An RL environment for a crowd route guidance using a crowd simulator.',
    url='https://github.com/ryonsd/crowdwalk-gym',
)
