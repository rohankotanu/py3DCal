from setuptools import setup, find_packages

setup(
    name='py3DCal',
    version='1.0.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy>=1.0.0',
        'matplotlib>=3.0.0',
        'pandas>=2.0.0',
        'scipy>=1.0.0',
        'torch>=2.0.0',
        'pyserial>=3.0',
        'opencv-python>=4.0.0',
        'pillow>=11.0.0',
        'tqdm>=4.0.0',
    ],
    entry_points={
        'console_scripts': [
            'list-com-ports = py3DCal:list_com_ports'
        ]
    }
)