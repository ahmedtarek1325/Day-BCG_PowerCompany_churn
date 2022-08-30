import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sb
import math 
def data_summary(data)->None:
    '''
    INPUT
    - This function takes a pandas data frame as an input 
    OUTPUT
    - It does not return 
    ACTIONS
    - This function does basic checks on the data frames 
        * Check duplicates 
        * Print data Info 
        * Print the data summay 
    '''
    
    print("Data shape is:",data.shape)
    print("The number of duplicates:",sum(data.duplicated()))
    print("The number of missing values:",sum(data.isna().sum()))
    print("\n\n")

    display(data.info())
    display(data.describe())


def __subplots(plotsNum):
    sqrtNum= math.sqrt(plotsNum)
    if  sqrtNum- int(sqrtNum)==0:
        rows,cols = sqrtNum,sqrtNum
    else:     
        rows,cols= int(sqrtNum),math.ceil(sqrtNum)+1
    return int(rows),int(cols)
def count_plots(data,x=[],y=[])->None: 
    '''
    INPUT
    - pandas dataframe
    - list of column names that we want to  do its count plot on x-axis
    - list of column names that we want to  do its count plot on y-axis
    OUTPUT
    - None
    ACTIONS
    - Plot the count plot for all passed data 
    '''
    color = sb.color_palette()[0]
    plots_num= len(x)+ len(y) 
    rows,cols= __subplots(plots_num)
    #plt.subplot(math.ceil(np.sqrt(numOfImgs)),math.ceil(np.sqrt(numOfImgs))+1,i+1)
    i= 1
    for col in x: 
        plt.subplot(rows,cols,i)
        plt.title(col +" Distributionion");
        if col in data.columns: sb.countplot(data=data,x=col,color= color);
        plt.xlabel("");

        i+=1
    for col in y: 
        plt.subplot(rows,cols,i)
        plt.title(col + " Distributionion");
        if col in data.columns: sb.countplot(data=data,y=col,color= color);
        i+=1
        plt.ylabel("");

    

