
def Outlier(x_data,y_data):

    import numpy as np
    from sklearn.svm import OneClassSVM as outlier_detector
    model=outlier_detector(kernel='rbf',)
    predicts=model.fit_predict(x_data,y_data)


    x_data2=[]
    y_data2=[]
    for i in range(len(predicts)):
        if predicts[i]==1:
            x_data2.append(x_data[i])
            y_data2.append(y_data[i])

    x_data=np.array(x_data2)
    y_data=np.array(y_data2)
    return x_data,y_data


