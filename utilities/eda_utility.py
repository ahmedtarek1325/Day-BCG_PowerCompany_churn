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
def count_plots(data,hue,x=[],y=[],subplots=None)->None: 
    '''
    INPUT
    - pandas dataframe
    - Hue in case of clustered plots
    - list of column names that we want to  do its count plot on x-axis
    - list of column names that we want to  do its count plot on y-axis
    - subplots order: to specify how many rows and columns 
    OUTPUT
    - None
    ACTIONS
    - Plot the count plot for all passed data 
    '''
    if not hue: color = sb.color_palette()[0] 
    else: color= None

    plots_num= len(x)+ len(y) 
    if subplots:
        rows,cols= subplots[0],subplots[1]
    else: 
        rows,cols= __subplots(plots_num)
    #plt.subplot(math.ceil(np.sqrt(numOfImgs)),math.ceil(np.sqrt(numOfImgs))+1,i+1)
    i= 1
    for col in x: 
        plt.subplot(rows,cols,i)
        plt.title(col +" Distributionion");
        if col in data.columns: sb.countplot(data=data,x=col,hue=hue,color= color);
        plt.xlabel("");

        i+=1
    for col in y: 
        plt.subplot(rows,cols,i)
        plt.title(col + " Distributionion");
        if col in data.columns: sb.countplot(data=data,y=col,hue=hue,color= color);
        i+=1
        plt.ylabel("");
    return 



def dsitribution_plots(data,x=[],bins_list=[[]],scales=[],titles=[],order=None)->None: 
    '''
    INPUT
    - pandas dataframe
    - list of column names that we want to make histogram plots for it 
    OUTPUT
    - None
    ACTIONS
    - Plot the histogram distribution plots for all passed data 
    '''
    if not order: 
        rows,cols= __subplots(len(x))
    else: 
        rows,cols= order[0],order[1] 
    
    i= 1
    for j,col in enumerate(x): 
        plt.subplot(rows,cols,i)
        plt.title(titles[j]);
        if col in data.columns:
            plt.hist(data= data, x= col, bins= bins_list[j] )
        plt.xscale(scales[j])
        plt.xlabel("");

        i+=1
    
    return




