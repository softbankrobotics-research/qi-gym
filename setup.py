import setuptools


with open("README.md", "r") as fh:
    readme = fh.read()
    readme = readme[:readme.index("<!-- start -->")] +\
        readme[(readme.index("<!-- end -->") + len("<!-- end -->")):]


setuptools.setup(
    name="qi-gym",
    version="0.0.1",
    author="Maxime Busy",
    author_email="",
    description="Reinforcment learning tools for SoftBank Robotics' robots",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/softbankrobotics-research/qi_gym",
    packages=setuptools.find_packages(),
    install_requires=['numpy', 'pybullet', 'qibullet', 'stable-baselines3'],
    package_data={"qi-gym": [
        "data/*.urdf",
        "data/*.mtl",
        "data/*.obj",
        "data/*.png"]},
    keywords=[
        'physics simulation',
        'robotics',
        'naoqi',
        'softbank',
        'pepper',
        'nao',
        'romeo',
        'robot',
        'reinforcment learning',
        'stable baselines 3'],
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft",
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ]
)
