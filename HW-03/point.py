# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 01:33:51 2020

@author: Sushant
"""
from math import sqrt

class point:
    x = 0
    y = 0
    def __init__(self,a,b):
        self.x = a
        self.y = b
        return None
    
    def getDistance(self):
        self.distance = sqrt((self.x**2) + (self.y**2))
        return self.distance
    
