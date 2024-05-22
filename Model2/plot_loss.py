# ast -> abstract syntax trees. 
# ast Module helps python applications to process trees of the python abstract syntax grammar.
import ast
import json

import matplotlib.pyplot as plt
import numpy as np

def moving_average(a,n=3):
    ret = np.cumsum(a,dtype=float)
    ret[n:] = ret[n:]-ret[:-n]
    return ret[n-1:]/n

# downloaded from google colab, after running our code there using GPU's
metrics_file = './metrics.json'

# metrics.json has the lines in str format. eg: '{'loss_val':, 'time':}'
# we are using ast.literal_eval() to convert the string to dictionary: {'loss_val':, 'time':}

with open(metrics_file,'r') as f:
    # metrics is a list of dict
    metrics = [ast.literal_eval(l[:-1]) for l in f.readlines()]
    f.close()

train_loss = []
for dict_ in metrics:
    if 'loss_box_reg' in dict_.keys():
        train_loss.append(float(dict_['loss_box_reg']))

val_loss = []
for dict_ in metrics:
    if 'val_loss_box_reg' in dict_.keys():
        val_loss.append(float(dict_['val_loss_box_reg']))


N = 40

train_loss_avg = moving_average(train_loss, n=N)
val_loss_avg = moving_average(val_loss, n=N)



# Plotting the train loss and val loss : Not very interpretable, hence we smoothen it for analysing
# Thus, we use Moving Average.

# plt.plot(range(0, 20 * len(train_loss), 20), train_loss, label='train loss')
# plt.plot(range(0, 20 * len(train_loss), 20), val_loss, label='val loss')


#  Plotting the Moving Average
plt.plot(range(20 * N - 1, 20 * len(train_loss), 20), train_loss_avg, label='train loss')
plt.plot(range(20 * N - 1, 20 * len(train_loss), 20), val_loss_avg, label='val loss')
plt.legend()
plt.grid()
plt.show()

# we need to look at these plots to decide, which of the model checkpoints (or model) to use.
        

# we can choose at 4000th check point
# ideally the training loss and the validation loss should be close to each other. (here it is not like that)