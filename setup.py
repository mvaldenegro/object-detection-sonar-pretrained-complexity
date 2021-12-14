import os
from setuptools import find_packages, setup


__version__ = "0.1"

if "VERSION" in os.environ:
    BUILD_NUMBER = os.environ["VERSION"].rsplit(".", 1)[-1]
else:
    BUILD_NUMBER = os.environ.get("BUILD_NUMBER", "dev")

dependencies = [
    "numpy",
    "opencv-python",
    "tensorflow==2.3.0",
    "pypaz",
    "matplotlib",
    ]

setup(
    name="marine_debris_detector",
    version="{0}.{1}".format(__version__, BUILD_NUMBER),
    description="A package to train marine debris detectors",
    author="Deepan Chakravarthi Padmanabhan, Matias Valdenegro-Toro",
    install_requires=dependencies,
    packages=find_packages(),
    zip_safe=False,
    entry_points=dict(
        console_scripts=[
            "marine_debris_detector=object_detection.trainer.train:train",
        ]
    ),
    data_files=[],
    python_requires=">=3.6,<=3.9",
)
