from utilities.FE_utilities import * 
from utilities.scores import * 
from utilities.models import *



def pipeline_for_cleaning(df):
    df= log_transfrom_columns(df)
    df= decide_peak_period(df,["date_end","date_renewal"])
    x= df.id
    drop_columns(df)
    df["id"]= x
    df= one_hot_encoding(df,catFeatures=["channel_sales","origin_up"])
    return df 



def simple_pipeline(df,sampling_type=1):
    '''
    make log transofrmation for skewed columns 
    extract peak periods 
    drop columns the non used columns 
    - split data 
    then make normalization 
    make upsampling OR Downsampling or Non depending on sampling_type
    '''
    df= log_transfrom_columns(df)
    df= decide_peak_period(df,["date_end","date_renewal"])
    drop_columns(df)
    df= one_hot_encoding(df,catFeatures=["channel_sales","origin_up"])
    X_train, X_test, y_train, y_test= split_x_y(df)
    
    X_train, X_test= normalize(X_train, X_test)
    
    
    X_train,y_train = sampling(X_train,y_train,sampling_type)

    
    return X_train, X_test, y_train, y_test

def simple_pipeline_cleaned_data(df,sampling_type=1):
    '''
    for cleaned ddataframes 
    - split 
    - normalize 
    - sampling
    '''
    
    X_train, X_test, y_train, y_test= split_x_y(df)    
    X_train, X_test= normalize(X_train, X_test)
    
    
    X_train,y_train = sampling(X_train,y_train,sampling_type)

    return X_train, X_test, y_train, y_test








def division_pipeline(df):
    drop_columns(df)
    df= one_hot_encoding(df,catFeatures=["channel_sales","origin_up"])
    
    
    
    X_train, X_test, y_train, y_test= split_x_y(df)
    X_train, X_test= normalize_min_max(X_train, X_test)

    return X_train, X_test, y_train, y_test
