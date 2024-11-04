from setuptools import setup, find_packages

# 从 pip list 输出生成依赖项列表
dependencies = [
    'numpy==1.26.4',
    'scanpy==1.10.2',
    'pandas==2.2.2',
    'timm==1.0.7',
    "opencv-contrib-python>=4.5.0",
    'scipy==1.13.1',
    'scikit-learn==1.5.1',
    'python-igraph==0.11.6'
]

setup(
    name='spotscope',
    version='0.1',
    author='Jiacheng Leng',
    author_email='amssljc@163.com',
    description='A description of your package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/my_package',
    packages=find_packages(),
    install_requires=dependencies,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
