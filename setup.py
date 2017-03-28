try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

install_requires = [
    'numpy>=1.10',
    'matplotlib>=2.0',
]

if __name__ == '__main__':
    setup(name='paradox',
          version="0.1",
          author='ictxiangxin',
          author_email='ictxiangxin@hotmail.com',
          maintainer='ictxiangxin',
          maintainer_email='ictxiangxin@hotmail.com',
          description='Tiny deep-learning framework',
          platforms=['MS Windows', 'Mac X', 'Unix/Linux'],
          keywords=['deep-learning', 'machine-learning', 'autograd', 'symbolic-computation'],
          packages=['paradox',
                    'paradox.kernel',
                    'paradox.neural_network',
                    'paradox.neural_network.convolutional_neural_network',
                    'paradox.utils',
                    'paradox.visualization',
                    'paradox.data_set'],
          install_requires=install_requires,
          classifiers=["Natural Language :: English",
                       "Programming Language :: Python",
                       "Operating System :: Microsoft :: Windows",
                       "Operating System :: Unix",
                       "Operating System :: MacOS",
                       "Programming Language :: Python :: 3"],
          zip_safe=False)
