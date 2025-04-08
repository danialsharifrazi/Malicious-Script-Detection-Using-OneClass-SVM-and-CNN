def NetPlot(net_histories,n_epch):
    import numpy as np
    import matplotlib.pyplot as plt
  
    losses=[]
    val_losses=[]
    accuracies=[]
    val_accuracies=[]

    for item in net_histories:
        
        history=item.history
        loss=history['loss']
        val_loss=history['val_loss']
        accuracy=history['accuracy']
        val_accuracy=history['val_accuracy']
        
        losses.append(loss)
        val_losses.append(val_loss)
        accuracies.append(accuracy)
        val_accuracies.append(val_accuracy)


    losses2=np.zeros((1,n_epch))
    val_losses2=np.zeros((1,n_epch))
    accuracies2=np.zeros((1,n_epch))
    val_accuracies2=np.zeros((1,n_epch))

    for i in losses:
        losses2+=i

    for i in val_losses:
        val_losses2+=i
    
    for i in accuracies:
        accuracies2+=i
    
    for i in val_accuracies:
        val_accuracies2+=i


    # 10 is number of folds
    losses2=(losses2/10).flatten()
    accuracies2=(accuracies2/10).flatten()
    val_losses2=(val_losses2/10).flatten()
    val_accuracies2=(val_accuracies2/10).flatten()

    plt.figure('Accracy Diagram_CNN_AUG',dpi=600)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(accuracies2,color='black',marker='o')
    plt.plot(val_accuracies2,color='green',marker='o')
    plt.ylim([0.70,1.0])
    plt.legend(['Train Data','Validation Data'],loc='lower right')
    plt.savefig('./proposed/Accuracy Diagram_CNN.jpg')

    plt.figure('Loss Diagram_CNN_AUG',dpi=600)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(losses2,color='black',marker='o')
    plt.plot(val_losses2,color='green',marker='o')
    plt.legend(['Train Data','Validation Data'],loc='upper right')
    plt.savefig('./proposed/Loss Diagram_CNN.jpg')



