# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 13:55:59 2018

@author: Reena
"""
############################Regression#########################################
import os
import pandas as pd
import numpy as np
os.chdir("F:/DATA SCIENTIST/Python/24_25Nov/")
trainData=pd.read_csv("R_Module_Day_7.2_Credit_Risk_Train_data.csv")
testData=pd.read_csv("R_Module_Day_8.2_Credit_Risk_Test_data.csv")

# create a column as called source for both train ad test data
trainData["Source"]="train"
testData["Source"]="test"

# dataset column names read
trainData.columns
testData.columns

# combind train and test data
combData=pd.concat([trainData,testData], axis=0)
combData.shape # confirm both data have been append one below the other
combData.head()
combData.describe()

# missing values check   NA values
combData.isnull().sum()
combData ["Gender"].isnull().sum()

# fill missing values

#combData ["Gender"] = combData["Gender"].fillna(combData ["Gender"].mode())
#cols_with_missing = (col for col in combData.columns if combData[col].isnull().any())
for i in list(combData):
    #if((i not in ['Loan ID','Load_Status','Source'])&(combData[i].isnull().sum()>0)):
      if( combData[i].dtype==object):
          tempimput=combData[i][combData.Source=='train'].mode()[0]
          combData[i].fillna(tempimput,inplace=True)
          # combData[i]=combData[i].fillna(combData.mode()[0])
           #combData[i]= combData[i].fillna(tempimput) 
      else:#(combData[i].dtype ==float64):
            tempimput=combData[i][combData.Source=='train'].median()
            combData[i].fillna(tempimput,inplace=True)#combData[:,i]=combData.iloc[:,i].fillna(combData.median())
         
combData.isnull().sum()

#dummy variable creation means:- categoery variable convert into continues values
combData.dtypes
dummies=pd.get_dummies(combData[['Gender','Married','Dependents','Education','Self_Employed', 'Property_Area']], drop_first=True, dtype=int)
dummies
            
CAT_VAR=['Gender','Married','Dependents','Education','Self_Employed', 'Property_Area']
#dummiesdf=pd.get_dummies(combData[CAT_VAR],drop_first =True)  #,drop_first =True                   
#dummiesdf.shape

# combind fd and dummey call in df2
df2=pd.concat([dummies,combData], axis = 1)
df2.columns

# df2, remove the origional all the catecorical variable stor into df3
droplist=['Gender','Married','Dependents','Education','Self_Employed','Property_Area']
#df3=pd.get_dummies(combData, columns=droplist, drop_first=True)
df3=df2.drop(['Gender','Married','Dependents','Education','Self_Employed','Property_Area'], axis=1) 
 
# df3 remove column loan_ID -->df4
#df4=df3.drop(combData.Loan_ID, axis=1)
#df5=df3.drop(['Loan_ID'], axis=1).copy()
df4=df3.drop(['Loan_ID'], axis=1) 
#df4=df3.drop(['Source'], axis=1) 
#df4.shape
#df4.columns
#origional show into df4 int,float,dummy_df,load_status,Source
# be ready df4
df4.shape
df4.columns

#df3=pd.get_dummies(combData, columns=droplist, drop_first=True, dtype=int64)
#trainData=df5.copy()
#df5=trainData.copy()
df4["Intercept"]=1
#trainData["Intercept"]=1
#testData["Intercept"]=1
#df4.groupby('Loan_Status').count()
df4['Loan_Status'].value_counts() # count number of Y and N values 
df4['Loan_Status'] = df4['Loan_Status'].map({'Y': 0, 'N': 1})  # replace Y to 1 and N to 0
#df5=trainData.copy()
#df4=df4.drop(['Loan_Status_Y'],axis=1)
#testData=testData.drop(['Source'],axis=1)

Train_temp = df4.loc[df4.Source == "train",:].drop('Source', axis = 1).copy()
Train_temp.shape
Test_temp = df4.loc[df4.Source == "test",:].drop('Source', axis = 1).copy()
Test_temp.shape

test_x=Test_temp.drop('Loan_Status',axis=1).copy()
test_Y=Test_temp['Loan_Status'].copy()
train_x=Train_temp.drop('Loan_Status',axis=1).copy()
train_Y=Train_temp['Loan_Status'].copy()
            
#build logistic model using statsmodels library
from statsmodels.api import Logit
M1=Logit(train_Y,train_x)
m1_Model=M1.fit()   # model build       
m1_Model.summary()  #model summary

# remove  highest signnificant variable one by one to improve the model              
Cols_To_Drop=['Dependents_3+']
M2=Logit(train_Y,train_x.drop(Cols_To_Drop,axis=1))            
M2Model=M2.fit()
M2Model.summary()
 
# Drop Self_Employed_Yes
Cols_To_Drop.append('Self_Employed_Yes')
M3 = Logit(train_Y, train_x.drop(Cols_To_Drop, axis = 1))
M3_Model = M3.fit()
M3_Model.summary()

Cols_To_Drop.append('Dependents_2')
M4=Logit(train_Y,train_x.drop(Cols_To_Drop,axis=1))            
M4Model=M4.fit()
M4Model.summary()

Cols_To_Drop.append('Property_Area_Urban')
M5=Logit(train_Y,train_x.drop(Cols_To_Drop,axis=1))            
M5Model=M5.fit()
M5Model.summary()

Cols_To_Drop.append('Loan_Amount_Term')
M6=Logit(train_Y,train_x.drop(Cols_To_Drop,axis=1))            
M6Model=M6.fit()
M6Model.summary()

Cols_To_Drop.append('Gender_Male')
M7=Logit(train_Y,train_x.drop(Cols_To_Drop,axis=1))            
M7Model=M7.fit()
M7Model.summary()


Cols_To_Drop.append('ApplicantIncome')
M8=Logit(train_Y,train_x.drop(Cols_To_Drop,axis=1))            
M8Model=M8.fit()
M8Model.summary()

Cols_To_Drop.append('LoanAmount')
M8l=Logit(train_Y,train_x.drop(Cols_To_Drop,axis=1))            
M8lModel=M8l.fit()
M8lModel.summary()

Cols_To_Drop.append('Education_Not Graduate')
M9=Logit(train_Y,train_x.drop(Cols_To_Drop,axis=1))            
M9Model=M9.fit()
M9Model.summary()

############predict on testset
columns_to_use=train_x.drop(Cols_To_Drop,axis=1).columns
test_x['test_prob']=M9Model.predict(test_x[columns_to_use])
test_x.columns

 
##########confusion matrix
confusionmat=pd.crosstab(test_x.test_class,test_Y)


















