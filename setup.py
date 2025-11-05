from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name='py3dcal',
    version='1.0.2',
    url="https://github.com/rohankotanu/py3DCal",
    author="Rohan Kota",
    author_email="rohankota2026@u.northwestern.edu",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
    install_requires=[
        'numpy>=1.0.0',
        'matplotlib>=3.0.0',
        'pandas>=2.0.0',
        'scipy>=1.0.0',
        'torch>=2.0.0',
        'torchvision>=0.23.0',
        'pyserial>=3.0',
        'opencv-python>=4.0.0',
        'pillow>=11.0.0',
        'tqdm>=4.0.0',
        'requests>=2.0.0',
        'scikit-learn>=1.0.0',
        'digit-interface>=0.2.1'
    ],
    entry_points={
        'console_scripts': [
            'list-com-ports = py3DCal:list_com_ports'
        ]
    }
)