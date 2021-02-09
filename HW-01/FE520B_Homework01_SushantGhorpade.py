#!/usr/bin/env python
# coding: utf-8

# # 1.Print

# In[4]:


# 1.Define a string variable,and print it.
a = 'Hello world'
print(a,'\n')
# 2.Define a string('I'm a Student),print it
b = "I'm a student"
print(b,'\n')
# 3.Defind a string:(How do you think of this course?Describe your feeling of this course)print it in multiple line
c = "How do you think of this course?\nDescribe your feeling of this course "
print(c)


# # 2.Operators

# In[34]:


# Define a = 100, b = 9, calculate following problems,
a = 100
b = 9
# 1. c = a + b, print c out.
c = a + b
print(c,'\n')
# 2. print the quotient of a/b.
print('The quotient of a/b.')
print((a/b),'\n')
# 3. print the integer part of a/b.
print('The integer part of a/b.')
print((a//b),'\n')
# 4. print the remainder part of a/b.
print('The remainder part of a/b.')
print(a%b,'\n')
# 5. print the result of ’a’ to the power of b.
print('The result of ’a’ to the power of b.')
print(a**b,'\n')
# 6. Using logic operator to return a Boolean value for a unequal to b.
print('Boolean value for a unequal to b')
print(a!=b,'\n')
# 7. Using logic operator to return a Boolean value for a greater than b.
print('Boolean value for a greater than b.')
print(a>b)


# # 3.List Practice

# In[35]:


# 1. Define a list Name it List A), whose items should include integer, float, and
# string. Please notice the length of the list should be greater than 5.

list_A = [10,62.75,"How are you doing?",20,22.5]
print(list_A,'\n')
print(len(list_A))


# In[29]:


# 2. Using extend and append to add another list(Name it List B) to List A.
list_A = [10,62.75,"How are you doing?",20,22.5]
list_B = [1,2,3]

#For Append
list_A.append(list_B)
print(list_A,'\n')


# In[31]:


#For Extend
list_A = [10,62.75,"How are you doing?",20,22.5]
list_B = [1,2,3]

list_A.extend(list_B)
print(list_A)


# In[33]:


# 3. Insert a string (’FE520’) to the second place of List A, and delete it after that.
list_A = [10,62.75,"How are you doing?",20,22.5]

list_A.insert(1,'FE520')
print(list_A)
print('\n')

#Delete from second place
del list_A[1]
print(list_A)


# In[46]:


# 4. Return and delete the last element in the List A, and print the new list.
list_A = [10,62.75,"How are you doing?",20,22.5]

del list_A[-1]
print(list_A)


# In[52]:


# 5. Return a new list (Name is List C), slicing the List A from 3rd to the end
list_A = [10,62.75,"How are you doing?",20,22.5]

list_C = list_A[2:]
print(list_C)


# In[53]:


# 6. Double size your List C
list_C = list_C*2
print(list_C)


# In[54]:


# 7. Reverse your sequence of List C.
print(list_C[::-1])


# # 4.Practice Dictionary

# In[55]:


# 1. Define a list A = [1, 2, 3, 5, 10, 1, 4, 10, 11, 20, 50, 100].
list_A = [1, 2, 3, 5, 10, 1, 4, 10, 11, 20, 50, 100]


# In[94]:


# 2. Write a loop to count the number of each unique digit into dictionary, where your
# keys are digit in the list A, and value is the count corresponding to each digit.
# Your result should look like :
# {1: 3, 2: 1, 3: 1, 5: 1, 10: 1, 4: 1, 11: 1, 20: 1, 50: 1, 100: 1}

x = set(list_A)
d = dict.fromkeys(x,0)

for i in d:
    count = 0
    for j in list_A:
        if i == j:
            count += 1
        
    d[i] = count
        
print(d)


# # 5.Loop Condition

# In[1]:


# Consider a sequenced list (or inversed sequence list, you need to consider an inversed
# sequence situation) and an inserted number. Define a function with two arguments, one
# is the sequenced list, another one is the inserted number. Please insert the number in
# the list with right place and output the new list. (Hint: you need to consider the special
# situation that the inserted number is smaller or greater than all numbers) Example:
# Input:
# List = [1, 2 , 4 , 9 , 17 , 25 , 63 ]
# InsertNum = 13
# Output:
# NewList = [1, 2 , 4 , 9 , 13 , 17 , 25 , 63 ]


# In[16]:



def NewList(List,num):   
    List.append(num)
    List.sort()
    return List

Input_List = []
    
user_input1 = int(input('Enter length of list required: '))
print('\n')

for i in range(0,user_input1):
    val = int(input('Enter value at position'+' '+str(i)+': '))
    Input_List.append(val)
       
num = int(input('\n''Enter number to add to list: '))
List = NewList(Input_List,num)
print(List)


# In[ ]:




