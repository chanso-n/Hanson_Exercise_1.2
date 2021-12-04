#import required libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import data
df_house = pd.read_csv('housing_data.csv')
df_mush = pd.read_csv('mushroom_data.csv')


#define RMSE function

def RMSE(sp, sp_pred):
#this ensures sp and sp_pred are np.arrays 
    sp = np.array(sp)
    sp_pred = np.array(sp_pred)
#squaring the difference of sale price and predicted sale price
    diffsq = (sp - sp_pred)**2
#finding the mean and sqrt to get the RMSE
    rmse = np.mean(diffsq)**(1/2)
    return rmse

print('The RMSE is:')
print(RMSE(df_house['sale_price'], df_house['sale_price_pred']))


#define MAE function

def MAE(sp, sp_pred):
#this ensures sp and sp_pred are np.arrays
    sp = np.array(sp)
    sp_pred = np.array(sp_pred)
#gets absolute values of difference of sale price and predicted sale price
    diffabs = np.abs(sp - sp_pred)
#finds the mean for diffabs
    mae = np.mean(diffabs)
    return mae

print('The MAE is:')
print(MAE(df_house['sale_price'], df_house['sale_price_pred']))


#define accuracy function

def acc(pred, act):
#get/set total number of array
    total = len(pred)
#find true postive predictions
    equal = [pred[i] == act[i] for i in range(len(pred))]
#sum of true positive predictions
    right = np.sum(equal)
#find accuracy by dividing right predictions by total    
    accuracy = right/total
    return accuracy

print('The accuracy score is:')
print(acc(df_mush['predicted'], df_mush['actual']))


#plot

p = np.linspace(0, 6)

#assigning the given equation to y

y = 0.005*(p**6) - 0.27*(p**5) + 5.998*(p**4) - 69.919*(p**3) + 449.17*(p**2) - 1499.7*(p) + 2028

#specifying the plot parameters
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
plt.ylim((0, 2.5))

#send to plot
plt.plot(p, y, 'r')

#show plot
plt.show()

print('Using the graph to estimate, the value of p that minimizes error is 5.172.')
print('Using the graph to estimate, the minimum error is 1.755.')
