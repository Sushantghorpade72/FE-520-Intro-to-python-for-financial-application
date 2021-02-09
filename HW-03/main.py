# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 01:11:34 2020

@author: Sushant
"""
#1 Class practice 30 pts
#1.1 Define class
#Create a class called Rectangular, with two data attributes: length and width. They
#should be assigned in constructor ( init ()) It should also have two function attributes
#called area() and perimeter() which return area and perimeter of this rectangular respectively


class Rectangular:
    lenght = 0
    width = 0
    def __init__(self,lenght,width):
        self.lenght = lenght
        self.width = width
        
    def area(self):
        return self.lenght*self.width
    
    def perimeter(self):
        return 2*(self.lenght+self.width)
    
#1.2 Numpy applying on class (10 pts)
#1. define two numpy array with size 10, named with length and width.
#2.test your your class Rectangular with input as np array. (Here you should have
#10 output for area and perimeter).

import numpy as np

lenght = np.array([1,2,3,4,5,6,7,8,9,10])
width = np.array([1,2,3,4,5,6,7,8,9,10])

myRec = Rectangular(lenght, width)
print('Area of rectangle\n',myRec.area())
print('\nPerimeter of rectangle\n',myRec.perimeter())


#2 Display Time (20pts)
#Create a Time class and initialize it with hours, minutes and seconds.
#1. Make a method addTime which should take two time object and add them. E.g.
#if your original initial parameter is (2 hour and 50 min and 10 seconds), then
#you could call this method with another input parameters (1 hr and 20 min and 5
#seconds) , then output is (4 hr and 10 min and 15 seconds)
#2. Make a method displayTime which should print the time (the initial parameter).
#3. Make a method DisplaySecond which should display the total seconds in the
#Time. E.g.- (1 hr 2 min) should display 3720 seconds


class Time:
    hours = 0
    minutes = 0
    seconds = 0
    def __init__(self,hours,minutes,seconds):
        self.total_seconds = (hours*60*60) + (minutes*60) + seconds
        seconds_to_minutes = self.total_seconds%3600
        self.hours = (self.total_seconds - seconds_to_minutes)/3600
        self.minutes = (seconds_to_minutes - self.seconds)/60
        self.seconds = (seconds_to_minutes)%60
        
    def addTime(self,time_obj):
        Total_Seconds = self.total_seconds + time_obj.total_seconds
        seconds_to_minutes = Total_Seconds%3600
        hrs = (Total_Seconds - seconds_to_minutes)/3600
        mins = (seconds_to_minutes - time_obj.secs)/60
        secs = (seconds_to_minutes)%60
        print('The output time is',hrs,'Hours',mins,'min',secs,'seconds')
        
    def displayTime(self):
        print('\nThe output time is',self.hours,'Hours',self.minutes,'min',self.seconds,'seconds')
        
    def DisplaySeconds(self):
        print('\nThe output time is',self.total_seconds,'seconds')
        
#test sample        
# t=Time(1,2,0)
# t.DisplaySeconds()