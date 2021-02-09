# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 14:42:03 2020

@author: Sushant
"""
import generator
import point
import numpy as np
import time

start = time.time()

rand_gen = generator.SCG(6,1103515245,12345,2**32) #tested/works with seed:2,6,14
rand_gen.startGenerator()

x= np.array(rand_gen.getList(10000000))
y= np.array(rand_gen.getList(10000000))

x = 2*(x-np.min(x))/np.ptp(x)-1
y = 2*(y-np.min(y))/np.ptp(y)-1

points = []
points_inside_circle = []

for i in range(np.size(x)):
    p = point.point(x[i],y[i])
    points.append(p)
    if p.getDistance()<1:
        points_inside_circle.append(p)
        
print('Ratio :',len(points_inside_circle)/len(points))
print('\nIt took',(time.time()-start),'seconds')
