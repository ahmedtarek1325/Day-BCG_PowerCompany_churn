from sklearn.ensemble import IsolationForest
import plotly.express as px 

def extract_outliers(data,cons_columns,col_name,contamination= 0.02):
    isf = IsolationForest(n_estimators = 100, random_state = 42, contamination =contamination)
    preds = isf.fit_predict(data[cons_columns])
    #extract outliers from the data
    col_name1= col_name +'_outliers'
    col_name2= col_name +'_outliers_scores'

    data[col_name1] = preds
    data[col_name1] = data[col_name1].astype(str)
    #extract the scores from the data in terms of strength of outlier
    data[col_name2] = isf.decision_function(data[cons_columns])
    print(data.groupby("churn")[col_name1].value_counts())
    return data

def scatter_plot(dataframe, x, y, color, title, hover_name):
    """Create a plotly express scatter plot with x and y values with a colour
    
    Input:
        dataframe: Dataframe containing columns for x, y, colour and hover_name data
        x: The column to go on the x axis
        y: Column name to go on the y axis
        color: Column name to specify colour
        title: Title for plot
        hover_name: column name for hover
        
    Returns:
        Scatter plot figure
    """
    #create the base scatter plot
    fig = px.scatter(dataframe, x = x, y=y,
                    color = color,
                     hover_name = hover_name)
    #set the layout conditions
    fig.update_layout(title = title,
                     title_x = 0.5)
    #show the figure
    fig.show()#create scatter plot