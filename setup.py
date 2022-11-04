from setuptools import setup, find_packages
from rayir.version import __version__

setup(
    name = "rayir",
    #install_requires = ['matplotlib', 'numpy'],
    #python_requires = '>=3',
    version = __version__,
    description = "Python 3 library.",
    author = ["Bilal Güngör", "Ahmet Anıl Dindar"],
    author_email = "bilalgungorr@gmail.com",
    packages = find_packages(),
    include_package_data=True,
    url = "http://github.com/bilalgungorr/rayir",
    entry_points = {
        'console_scripts': [],
    },
    keywords = ["Moment-curvature", "P-M interaction", "RC Section Design"],
    classifiers = [
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        #"Operating System :: Microsoft :: Windows",
        #"Operating System :: POSIX",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering"],
    long_description = ""
   )
