from setuptools import setup, find_packages

setup(
    name='ai2thor-rearrangement',
    description="Fork of repo. Enables simple use of ai2thor environment for meta-rl agents",
    author="allenai",
    packages=find_packages(),
    exclude=["data", "pretrained_model_ckpts"],
    install_requires=[
        "ai2thor>=2.7.2,!=2.8.0",
        "allenact>=0.4.0",
        "allenact_plugins[ithor]>=0.4.0",
        "numpy",
        "torch>1.6.0",
        "torchvision>=0.7.0",
        "matplotlib",
        "scipy",
        "stringcase",
        "lru-dict",
        "networkx",
        "compress-pickle==1.2.0",
        "packaging",
    ]
)