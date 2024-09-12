def FSelection(x_data,y_data):
    import numpy as np    

    from sklearn.feature_selection import VarianceThreshold
    Var=0.8*(1-0.8)
    feature_selector=VarianceThreshold(threshold=Var)
    x_data=feature_selector.fit_transform(x_data,y_data)

    return x_data,y_data











