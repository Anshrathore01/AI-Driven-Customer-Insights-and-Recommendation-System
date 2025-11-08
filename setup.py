from setuptools import find_packages,setup
from typing import List
HYPHEN_E_DOT='-e .'
def get_requirements(file_path : str)-> List[str]:
    ##this function will fetch requirements from requirements.txt
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req .replace("/n"," ") for req in requirements]
        
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements

setup(
    name="AI-Driven-Customer-Insights-and-Recommendation-System",
    version='0.0.1',
    author="Ansh Rathore",
    author_email="anshthakur258045@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)