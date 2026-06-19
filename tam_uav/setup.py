from setuptools import find_packages, setup


setup(
    name="tam-uav",
    version="0.1.0",
    description="TAM-HAPPO aligned F22 MAV/F16 UAV direct-control air combat environment",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.23",
        "PyYAML>=6.0",
        "torch>=2.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0"],
        "gym": ["gymnasium>=0.28"],
    },
)
