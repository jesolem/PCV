#!/usr/bin/env python

from distutils.core import setup

setup(name='PCV',
        version='1.0',
        author='Jan Erik Solem',
        url='https://github.com/jesolem/PCV',
        packages=['PCV', 'PCV.classifiers', 'PCV.clustering', 'PCV.geometry', 
                'PCV.imagesearch', 'PCV.localdescriptors', 'PCV.tools'],
        requires=['NumPy', 'Matplotlib', 'SciPy'],
        )
