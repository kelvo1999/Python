#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 15:13:11 2021

@author: warthog
"""


import numpy as np
import matplotlib.pyplot as plt



np.random.seed(0)

x = 2 - 3 * np.random.normal(0, 1, 20)

y = x - 2 * (x ** 2) + 0.5 * (x ** 3) + np.random.normal(-3, 3, 20)

plt.scatter(x,y, s=10)

plt.show()
