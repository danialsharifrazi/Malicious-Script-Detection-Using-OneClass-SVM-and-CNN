import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from OutlierDetection import Outlier
import datetime
from sklearn.metrics import auc, classification_report,confusion_matrix, roc_curve
from sklearn.model_selection import KFold, train_test_split
from keras.callbacks import CSVLogger
from keras.layers import Conv1D,Dense, Dropout, Flatten
from keras.models import Sequential
from keras.utils import np_utils
from keras.callbacks import CSVLogger
import PlotHistory_kFold


path='./Dataset.csv'
dataset=pd.read_csv(path)
df=pd.DataFrame(dataset)
labels=df['label']
labels=list(labels)


data1=[]
data2=[]
label1=[]
label2=[]
for i in range(len(labels)):
    if labels[i]=='ben':
        data1.append(df.iloc[i])
        label1.append(0)
    else:
        data2.append(df.iloc[i])
        label2.append(1)

data1.extend(data2)
x=np.array(data1)
x_data=x[:,1:]
x_data=x_data.astype('float32')
label1.extend(label2)
y_data=np.array(label1)

print('shape before outlier: ',x_data.shape,y_data.shape)

# Scaling
x_data,y_data=Outlier(x_data, y_data)
scaler1=MinMaxScaler()
x_data=scaler1.fit_transform(x_data)


x_data=x_data.reshape((x_data.shape[0],x_data.shape[1],1))
print('shape after outlier: ',x_data.shape,y_data.shape)

lst_loss=[]
lst_acc=[]
lst_reports=[]
lst_AUC=[]
lst_matrix=[]
lst_times=[]
lst_history=[]
fold_number=1
n_epch=50

kfold=KFold(n_splits=10,shuffle=True,random_state=None)
for train,test in kfold.split(x_data,y_data):

    x_train=x_data[train]
    x_test=x_data[test]
    y_train=y_data[train]
    y_test=y_data[test]

    x_train,x_valid,y_train,y_valid=train_test_split(x_train,y_train,test_size=0.2,random_state=0)

    print(f'train: {x_train.shape}  {y_train.shape}')
    print(f'test: {x_test.shape}  {y_test.shape}')
    print(f'valid: {x_valid.shape}  {y_valid.shape}')


    calback=CSVLogger(f'./results/CNN/logger_fold{fold_number}.log')

    y_train=np_utils.to_categorical(y_train)
    y_test=np_utils.to_categorical(y_test)
    y_valid=np_utils.to_categorical(y_valid)


    model=Sequential()
    model.add(Conv1D(32,3,padding='same',activation='relu',strides=2,input_shape=(x_train.shape[1],x_train.shape[2])))
    model.add(Conv1D(64,3,padding='same',activation='relu',strides=2))
    model.add(Conv1D(128,3,padding='same',activation='relu',strides=2))
    model.add(Flatten())
    model.add(Dense(64,activation='relu'))
    model.add(Dense(32,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2,activation='sigmoid'))

    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
            
    start=datetime.datetime.now()
    net_history=model.fit(x_train, y_train, batch_size=128, epochs=n_epch,validation_data=(x_valid,y_valid),callbacks=[calback])
    end=datetime.datetime.now()
    training_time=end-start

    model.save(f'./results/CNN/CNN_fold{fold_number}.h5')

    test_loss, test_acc=model.evaluate(x_test,y_test)

    predicts=model.predict(x_test)
    predicts=predicts.argmax(axis=1)
    actuals=y_test.argmax(axis=1)

    fpr,tpr,_=roc_curve(actuals,predicts)
    a=auc(fpr,tpr)
    r=classification_report(actuals,predicts)
    c=confusion_matrix(actuals,predicts)


    lst_history.append(net_history)
    lst_times.append(training_time)
    lst_acc.append(test_acc)
    lst_loss.append(test_loss)
    lst_AUC.append(a)
    lst_reports.append(r)
    lst_matrix.append(c)

    fold_number+=1

        

PlotHistory_kFold.NetPlot(lst_history,n_epch)

path=f'./results/CNN/CNN_Results.txt' 
f1=open(path,'a')
f1.write('Average accuracy: '+str(np.mean(lst_acc)))
f1.write('\nAccuracies: '+str(lst_acc)+'\nLosses: '+str(lst_loss)+'\nAUCs: '+str(lst_AUC)+'\n')
f1.write('\n\nMetrics for all Folds: \n\n')
for i in range(len(lst_reports)):
    f1.write(str(lst_reports[i]))
    f1.write('\n\nTraining Time: '+str(lst_times[i]))
    f1.write('\n\nCofusion Matrix: \n'+str(lst_matrix[i])+'\n\n______________________\n')
f1.close()



