import numpy as np
import soft_fit_and_plot as sof
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 
from scipy.interpolate import griddata #interpoláció


x = np.linspace(1,5,100)
y = np.linspace(0,4,100)


X, Y = np.meshgrid(x, y)

data = sof.f(X,Y,10,3)#ez a függvényünk

#ez = np.vstack((x,y))

sof.plot2d(data,x,y)

print(X.shape)
plt.scatter(X,Y,data*10)
plt.show()

##### ez az átintepolálás:
x_new = np.linspace(1,5,10)
y_new = np.linspace(0,4,10)

X_new,Y_new =np.meshgrid(x_new,y_new)

f_gridded = griddata((X.flatten(),Y.flatten()),data.flatten(),(X_new,Y_new),method = 'linear')

sof.plot2d(f_gridded,X_new,Y_new)
####################
#####ez most az illesztés:

param = sof.illeszto(X,Y,data) #jól működik az eredetre

print('eredetire:',param)

param = sof.illeszto(X_new,Y_new,f_gridded) #illesztésre

print('át gridezetre:',param)

