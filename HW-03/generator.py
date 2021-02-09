# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 01:24:43 2020

@author: Sushant
"""
#Q. create LGC class
class LCG:
    seed = 1
    multiplier = 1
    increment = 1
    modulus = 1
    def __init__(self,seed,multiplier,increment,modulus):
        self.seed = seed
        self.multiplier = multiplier
        self.increment = increment
        self.modulus = modulus
    
    #get the seed    
    def getSeed(self):
        return self.seed
    
    #set the seed
    def setSeed(self,NewSeed):
        self.seed = NewSeed
        return NewSeed
    
    #initialize the generator
    def startGenerator(self):
        self.gen = (self.seed*self.multiplier + self.increment)%self.modulus
        return self.gen
    
    #get next random number
    def getRandn(self):
        if self.gen:
            self.gen = (self.gen*self.multiplier + self.increment)%self.modulus
        else:
            self.gen = (self.gen*self.seed + self.increment)%self.modulus
        return self.gen
    
    #get list of random numbers
    def getList(self,count):
        randnList = []
        for i in range(count):
            if self.gen:
                self.gen = (self.gen*self.multiplier + self.increment)%self.modulus
            else:
                self.gen = (self.gen*self.seed + self.increment)%self.modulus
            randnList.append(self.gen)
        return randnList 
    
#Create an inherited class called SCG from class LCG you have created

class SCG(LCG):
    seed = 1
    multiplier = 1
    increment = 1
    modulus = 1
    def __init__(self,seed,multiplier,increment,modulus):
        if seed%4!=2:
            print('Seed doesnot satisfy given condition, enter valid seed value')
            return None
        self.seed = seed
        self.multiplier = multiplier
        self.increment = increment
        self.modulus = modulus
        
    def getSeed(self):
        return self.seed
    
    def setSeed(self,NewSeed):
        self.seed = NewSeed
        return NewSeed
    
    def startGenerator(self):
        self.gen = (self.seed*self.multiplier + self.increment)%self.modulus
        return self.gen
    
    def getRandn(self):
        if self.gen:
            self.gen = (self.gen*self.multiplier + self.increment)%self.modulus
        else:
            self.gen = (self.gen*self.seed + self.increment)%self.modulus
        return self.gen
    
    def getList(self,count):
        randnList = []
        for i in range(count):
            if self.gen:
                self.gen = (self.gen*self.multiplier + self.increment)%self.modulus
            else:
                self.gen = (self.gen*self.seed + self.increment)%self.modulus
            randnList.append(self.gen)
        return randnList 
    
