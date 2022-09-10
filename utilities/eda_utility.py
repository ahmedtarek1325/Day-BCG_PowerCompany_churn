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



'''
#create a function to calculate IQR bounds
def IQR_bounds(dataframe, column_name, multiple):
    """Extract the upper and lower bound for outlier detection using IQR
    
    Input:
        dataframe: Dataframe you want to extract the upper and lower bound from
        column_name: column name you want to extract upper and lower bound for
        multiple: The multiple to use to extract this
        
    Output:
        lower_bound = lower bound for column
        upper_bound = upper bound for column"""
    
    #extract the quantiles for the column
    lower_quantile = dataframe[column_name].quantile(0.25)
    upper_quantile = dataframe[column_name].quantile(0.75)
    #cauclat IQR
    IQR = upper_quantile - lower_quantile
    
    #extract lower and upper bound
    lower_bound = lower_quantile - multiple * IQR
    upper_bound = upper_quantile + multiple * IQR
    
    #retrun these values
    return lower_bound, upper_bound
#set the columns we want
columns = ["attack", "defense"]
#create a dictionary to store the bounds
column_bounds = {}
#iteratre over each column to extract bounds
for column in columns:
    #extract normal and extreme bounds
    lower_bound, upper_bound =  IQR_bounds(pokemon, column, 1.5)
    #send them to the dictionary
    column_bounds[column] = [lower_bound, upper_bound]
#create the normal dataframe
pokemon_IQR_AD = pokemon[(pokemon["attack"] < column_bounds["attack"][0]) | 
                         (pokemon["attack"] > column_bounds["attack"][1]) |
                         (pokemon["defense"] < column_bounds["defense"][0]) | 
                         (pokemon["defense"] > column_bounds["defense"][1])
                        ]#create a function to calculate IQR bounds
def IQR_bounds(dataframe, column_name, multiple):
    """Extract the upper and lower bound for outlier detection using IQR
    
    Input:
        dataframe: Dataframe you want to extract the upper and lower bound from
        column_name: column name you want to extract upper and lower bound for
        multiple: The multiple to use to extract this
        
    Output:
        lower_bound = lower bound for column
        upper_bound = upper bound for column"""
    
    #extract the quantiles for the column
    lower_quantile = dataframe[column_name].quantile(0.25)
    upper_quantile = dataframe[column_name].quantile(0.75)
    #cauclat IQR
    IQR = upper_quantile - lower_quantile
    
    #extract lower and upper bound
    lower_bound = lower_quantile - multiple * IQR
    upper_bound = upper_quantile + multiple * IQR
    
    #retrun these values
    return lower_bound, upper_bound
#set the columns we want
columns = ["attack", "defense"]
#create a dictionary to store the bounds
column_bounds = {}
#iteratre over each column to extract bounds
for column in columns:
    #extract normal and extreme bounds
    lower_bound, upper_bound =  IQR_bounds(pokemon, column, 1.5)
    #send them to the dictionary
    column_bounds[column] = [lower_bound, upper_bound]
#create the normal dataframe
pokemon_IQR_AD = pokemon[(pokemon["attack"] < column_bounds["attack"][0]) | 
                         (pokemon["attack"] > column_bounds["attack"][1]) |
                         (pokemon["defense"] < column_bounds["defense"][0]) | 
                         (pokemon["defense"] > column_bounds["defense"][1])
                        ]
'''

'''
from keras import layers
from keras import models
from keras import optimizers
from keras import losses
from keras import regularizers
from keras import metrics
# add validation dataset
validation_split=1000
x_validation=x_train[:validation_split]
x_partial_train=x_train[validation_split:]
y_validation=y_train[:validation_split]
y_partial_train=y_train[validation_split:]
model=models.Sequential()
model.add(layers.Dense(8,kernel_regularizer=regularizers.l2(0.003),activation='relu',input_shape=(10000,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(8,kernel_regularizer=regularizers.l2(0.003),activation='relu'))
model.add(layers.Dropout(0.6))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x_partial_train,y_partial_train,epochs=4,batch_size=512,validation_data=(x_validation,y_validation))
print("score on test: " + str(model.evaluate(x_test,y_test)[1]))
print("score on train: "+ str(model.evaluate(x_train,y_train)[1]))


'''
