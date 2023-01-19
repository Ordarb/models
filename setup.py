import os

from setuptools import setup, find_packages

path = os.path.dirname(os.path.realpath(__file__))
requirements_path = path + '/requirements.txt'
install_requires = []

if os.path.isfile(requirements_path):
    with open(requirements_path) as f:
        lst = f.read().splitlines()

    # filter ssh git
    # https://stackoverflow.com/questions/32688688/how-to-write-setup-py-to-include-a-git-repository-as-a-dependency
    for x in lst:
        if '+ssh' in x:
            repo_name = x[x.rfind('/') + 1:x.rfind('.git')]
            path = f'{repo_name} @ {x}'
            install_requires.append(path)
        else:
            install_requires.append(x)

setup(name='models',
      version='0.0',
      packages=find_packages(),
      package_data={"": ["*.yaml", ".txt", "*.xlsx", "*.xls", "*.csv", "*.png", "*.jpeg", "*.jpg"]},
      install_requires=install_requires
      )