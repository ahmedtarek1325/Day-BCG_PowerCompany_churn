from datetime import date
from re import X
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

dropping_columns= ["id","date_activ","date_modif_prod","forecast_discount_energy","forecast_price_energy_peak",
                    "forecast_cons_year","has_gas","margin_net_pow_ele","cons_last_month"]


log_transform_columns= ["cons_12m","cons_gas_12m", "margin_gross_pow_ele","net_margin",
                        "imp_cons","pow_max",'forecast_cons_year','forecast_meter_rent_12m'] 

target_variable= "churn"
log_smooth = 1 

def drop_columns(df): 
    """
    INPUT
    - Dataframe we want to reduce its features
    OUTPUT
    - None
    ACTIONS
    - Drop columns in the list of dropping_columns from the dataframe 
    """
    for col in dropping_columns: 
        if col in df.columns: 
            df.drop(columns=[col],inplace=True)
    


    return

def log_transfrom_columns(df):
    """
    INPUT
    - Dataframe 
    OUTPUT
    - Dataframe 
    ACTIONS
    -make log transformation for all the specified axes in log_transform_columns 
    if they are in the dataframe
    """
    for col in log_transform_columns: 
        if col in df.columns: 
            df[col] = np.log10(df[col]+log_smooth)
    return df 
    


def one_hot_encoding(df,catFeatures=[],min_fetures=20):
    '''
    INPUTS
    - dataframe
    - min number of instances taht should be in a certain column
    OUTPUTS
    - modified Dataframe
    ACTIONS
    - apply get dummies then
    - remove features that has values_counts < min
    - remove MISSING column if exist otherwise remove any column
    '''
    for col in catFeatures:
        # categories in the cols that are smaller than min_features
        cols_to_drop= df[col].value_counts().index[ df[col].value_counts() < min_fetures] 
        cols_to_drop= list(map((lambda f: col+ '_'+ f),list(cols_to_drop)))
        df = pd.get_dummies(df, columns = [col])
        df.drop(columns= cols_to_drop,inplace=True)
        
        df.drop(columns=[col+'_MISSING'],inplace=True)
    return df
        
    
    
def split_x_y(df,splitsize=0.15,random_state=42): 
    '''
    INPUT 
    - Takes the dataframe
    - split size
    - randomstate
    OUTPUT
    dataframes
    - X_train, X_test, y_train, y_test
    '''
    target= df[target_variable]
   
    X_train, X_test, y_train, y_test = train_test_split( df.drop(columns=[target_variable]),target,
                                                        test_size=splitsize, random_state=random_state)
    return X_train, X_test, y_train, y_test


def normalize(X_train,X_test):
    '''
    INPUT 
    - takes numercal features for both X_train,and  X_test

    OUTPUT 
    - numpydarray of Normalized X_train,and  X_test
    ACTIONS
    - Fit on X_train then
    - Transform on both X_train and X_test
    '''
    scaler= StandardScaler()
    cols=[]
    for col in X_train.columns: 
        if len(X_train[col].value_counts())> 10 and X_train[col].dtype != object: 
            cols.append(col)
    
    scaler.fit(X_train[cols])
    

    X_train[cols]= scaler.transform(X_train[cols])
    X_test[cols]= scaler.transform(X_test[cols])
    
    return X_train,X_test

def __decide_period(date_):
    '''
    INPUT 
    - datetime: pandas series
    OUTPUT
    - str: decides if its peak,off peak or others 
    ACTIONS 
    - the first three conditions checks if its summer if so then its peak
    - The second three conditions check if its winter if so then its offpeak 
    - Otherwise then its a none
    '''
    month,day= date_.month,date_.day
    if month==7 or month == 8: 
        return "peak"
    elif month==6 and day>= 20:
        return "peak"
    elif month==9 and day<= 23:
        return "peak"
    elif month==1 or month==2:
        return "offpeak"
    elif month==12 and day >=21:
        return "offpeak"
    elif month ==3 and day<=21: 
        return "offpeak"
    return "other" 

def decide_peak_period(df,dates_columns=[]):
    if not dates_columns:
        return df 
	
    for col in dates_columns: 
        df[col]= pd.to_datetime(df[col])
        df[col]=df[col].apply(__decide_period)
        df = pd.get_dummies(df, columns = [col])
        df.drop(columns= [col+"_other"],inplace=True)
    
    return df         


def sampling(X_train,y_train,samplingType=0):
    '''
    INPUT 
    - X_train: pandas or numpy 
    - y_train: pandas or numpy
    - sampling type : int
        * 0 : no sampling 
        * 1 : upsampling with smote
        * 2 : Undersampling Then Upsampling
    OUTPUT 
    - X_train : pandas or numpy depends on the input
    - Y_train : the same as above
    '''


    if samplingType==1: 
        smote = SMOTE(random_state=42)
        X_train, y_train= smote.fit_resample(X_train,y_train )
    elif samplingType==2: 
        rus = RandomUnderSampler(random_state=42) 
        smote = SMOTE(sampling_strategy=0.8,random_state=42)
        
        X_train, y_train= smote.fit_resample(X_train,y_train )
        X_train, y_train= rus.fit_resample(X_train,y_train )
    
    return  X_train, y_train