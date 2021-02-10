# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 18:39:37 2020

@author: Sushant
"""
#Import packages
import pandas as pd
import numpy as np
import re

#Q1.Credit Transaction data
print('\n---Q1.Credit Transaction data---\n')
ctdf = pd.read_csv('res_purchase_2014.csv')                           #Import file
print('\n---> There are',ctdf.shape[0],'rows and',ctdf.shape[1],'columns\n')

#Define a function to clean 'Amount' Column
def clean_amount(amt):
    if '$' in str(amt):
        amt = amt.replace('$','')
    if '(' in str(amt):
        amt = amt.replace('(','-')
    if ')' in str(amt):
        amt = amt.replace(')','')
    amt = re.sub("[^-.0-9]", "", str(amt))
    return float(amt)

#Apply function on 'Amount'
ctdf['Amount'] = ctdf.Amount.apply(clean_amount)

#1. What is total amount spending captured in this dataset?
print('\n---> Total amount spending captured in this dataset is',round(sum(ctdf['Amount']),2),'dollars\n')

#2. How much was spend at WW GRAINGER? Hint: All ‘WW GRAINGER’ contained in the ‘Vendor’.
garinger = ctdf.loc[ctdf['Vendor']=='WW GRAINGER']
print('\n---> Amount spend at "WW GRAINGER" is',round(sum(garinger['Amount']),2),'dollars\n')

#3.How much was spend at WM SUPERCENTER? Hint: All ‘WM SUPERCENTER’ contained in the ‘Vendor’.
supercenter = ctdf.loc[ctdf['Vendor']=='WM SUPERCENTER']
print('\n---> Amount spend at "WM SUPERCENTER" is',round(sum(supercenter['Amount']),2),'dollars\n')

#4.How much was spend at GROCERY STORES? Hint: All ‘GROCERY STORES’ contained in the ‘Merchant Category Code’.
grocery_stores = ctdf.loc[ctdf['Merchant Category Code (MCC)']=='GROCERY STORES,AND SUPERMARKETS']
print('\n---> Amount spend at "GROCERY STORES" is',round(sum(grocery_stores['Amount']),2),'dollars\n')



print('\n---Q2.Data Processing with Pandas---\n')
#2.Data Processing with Pandas (60 points)
BalanceSheet = pd.read_excel('Energy.xlsx') #import as Balancesheet
Ratings = pd.read_excel('EnergyRating.xlsx') #import as ratings

print('\n---> BalanceSheet dataset has',BalanceSheet.shape[0],'rows and',BalanceSheet.shape[1],'columns')
print('\n---> Ratings dataset has',Ratings.shape[0],'rows and',Ratings.shape[1],'columns\n')

#2. drop the column if more than 90% value in this colnmn is 0 (or missing value).
#3. replace all None or NaN with average value of each column.

BalanceSheet_clean = BalanceSheet.replace([0,' ','NULL'],np.mean) #Replacing by mean
BalanceSheet_clean = BalanceSheet_clean.dropna(thresh= BalanceSheet_clean.shape[0]*0.9,how='all',axis=1) #Drop columns with 90% missing values
print('---> After dropping columns with more than 90% missing values,BalanceSheet dataset has',BalanceSheet_clean.shape[0],'rows and',BalanceSheet_clean.shape[1],'columns')

Ratings_clean = Ratings.replace([0,' ','NULL'],np.mean) #Replacing by mean
Ratings_clean = Ratings_clean.dropna(thresh= Ratings.shape[0]*0.9 == 0,how='all',axis=1) #Drop columns with 90% missing values
print('\n---> After dropping columns with more than 90% missing values,Ratings dataset has',Ratings_clean.shape[0],'rows and',Ratings_clean.shape[1],'columns\n')

# #Define function to normalize numeric data
def normalize(x,column_min,column_max):
    if column_max == column_min:
        return x
    x = x-column_min/(column_max-column_min)
    return x


#Apply Normalization formula
#For BalanceSheet
for feature in BalanceSheet_clean.select_dtypes(include=[np.number]).columns:
    BalanceSheet_clean[feature] = BalanceSheet_clean[feature].apply(normalize,args=(BalanceSheet_clean[feature].min(),BalanceSheet_clean[feature].max()))

#For Ratings
for feature in Ratings_clean.select_dtypes(include=[np.number]).columns:
    Ratings_clean[feature] = Ratings_clean[feature].apply(normalize,args=(Ratings_clean[feature].min(),Ratings_clean[feature].max()))   

# # 5. Define an apply function to return the statistical information for variables = [’Current Assets - Other - Total’, ’Current Assets - Total’, ’Other Long-term
# # Assets’, ’Assets Netting & Other Adjustments’], you need to return a dataframe which has exactly same format with pandas method .describe().
def get_stats(df):
    datadict={'index':['count','mean','std','min','25%','50%','75%','max'],'Current Assets - Other - Total':[df['Current Assets - Other - Total'].count(),df['Current Assets - Other - Total'].mean(),df['Current Assets - Other - Total'].std(),df['Current Assets - Other - Total'].min(),df['Current Assets - Other - Total'].quantile(0.25),df['Current Assets - Other - Total'].quantile(0.5),df['Current Assets - Other - Total'].quantile(0.75),df['Current Assets - Other - Total'].max()],'Current Assets - Total':[df['Current Assets - Total'].count(),df['Current Assets - Total'].mean(),df['Current Assets - Total'].std(),df['Current Assets - Total'].min(),df['Current Assets - Total'].quantile(0.25),df['Current Assets - Total'].quantile(0.5),df['Current Assets - Total'].quantile(0.75),df['Current Assets - Total'].max()],'Other Long-term Assets':[df['Other Long-term Assets'].count(),df['Other Long-term Assets'].mean(),df['Other Long-term Assets'].std(),df['Other Long-term Assets'].min(),df['Other Long-term Assets'].quantile(0.25),df['Other Long-term Assets'].quantile(0.5),df['Other Long-term Assets'].quantile(0.75),df['Other Long-term Assets'].max()],'Assets Netting & Other Adjustments':[df['Assets Netting & Other Adjustments'].count(),df['Assets Netting & Other Adjustments'].mean(),df['Assets Netting & Other Adjustments'].std(),df['Assets Netting & Other Adjustments'].min(),df['Assets Netting & Other Adjustments'].quantile(0.25),df['Assets Netting & Other Adjustments'].quantile(0.5),df['Assets Netting & Other Adjustments'].quantile(0.75),df['Assets Netting & Other Adjustments'].max()]}
    new_df = pd.DataFrame(datadict)
    new_df = new_df.set_index('index')
    return new_df

#BalanceSheet.describe()
stats = get_stats(BalanceSheet)
print('\n',stats)

#6.find Correlation matrix
corr_variables = BalanceSheet[['Current Assets - Other - Total','Current Assets - Total', 'Other Long-term Assets', 'Assets Netting & Other Adjustments']]
correlation = corr_variables.corr()
print('\n',correlation)

# 7. If you look at column (’Company Name’), you will find some company name end with ’CORP’, ’CO’ or ’INC’. Create a new column (Name: ’CO’) to store
# the last word of company name. (For example: ’CORP’ or, ’CO’ or ’INC’) (Hint:using map function)
BalanceSheet_clean['CO'] = BalanceSheet_clean['Company Name'].str.split(' ').apply(lambda x:x[-1])
#print(BalanceSheet_clean['CO'].head())
#print(BalanceSheet_clean['CO'].value_counts())

#8.Merge (inner) Ratings and BalanceSheet based on ’datadate’ and ’Global Company Key’, and name merged dataset ’Matched’.
Matched = pd.merge(BalanceSheet_clean,Ratings_clean,on=['Data Date','Global Company Key'],how='inner')

#9. Mapping
match_map = {'AAA':0,'AA+':1,'AA':2,'AA-':3,'A+':4,'A':5,'A-':6,'BBB+':7,
                                        'BBB':8,'BBB-':9,'BB+':10,'BB':11,'others':12}

Matched['Rate'] = Matched['S&P Domestic Long Term Issuer Credit Rating'].map(match_map)            
#print(Matched['Rate'].head())

#10. Calculate the rating frequency of company whose name end with ’CO’. 
# (Calculate the distribution of rating given the company name ending with ’CO’, Hint,use map function)
#Creating seprate data for Rate and CO
dfCompany = Matched[['Rate','CO']].copy()
dfCompany = dfCompany[dfCompany['CO']=='CO']
#For frequency rating of companies
freq_rate = dfCompany.groupby('Rate').count()
print('\n',freq_rate)
