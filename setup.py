from setuptools import setup, find_packages
setup(
    name = 'MixGate', 
    version = '2.0.1',
    description= 'MixGate: Multi-view Representation Learning for Netlists', 
    author = 'MixGate', 
    author_email= 's1114263143@163.com', 
    packages = find_packages(exclude=['examples', 'data']),
    include_package_data = True,
)