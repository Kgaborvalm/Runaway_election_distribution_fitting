import numpy as np
import matplotlib.pyplot as plt
import h5py
#import matplotlib #colorbarhoz
from scipy.optimize import curve_fit 
from scipy.interpolate import griddata #interpoláció

def plot2dpolar(data,r_grid=None,theta_grid=None):
    print("ezaaz")
    R,Th = np.meshgrid(r_grid,theta_grid)
    X = R*np.cos(Th)
    Y = R*np.sin(Th)

    


    #data=f(X,Y,1,1)
    #data = np.ones(data.shape)
    plt.figure()
    plot = plt.pcolormesh(X, Y, data,cmap='inferno',shading='auto')
    cbar = plt.colorbar(plot)
    #cbar.formatter = matplotlib.ticker.LogFormatterExponent(base=10)
    #cbar.update_ticks()

    #plt.show()

def plot2d(data,x,y):
    #X,Y =np.meshgrid(x,y)
    
    plt.figure()
    plot = plt.pcolormesh(x, y, data,cmap='inferno',shading='auto')
    cbar = plt.colorbar(plot)
    plt.show()



def illeszto(x,y,z):
    #x,y = np.meshgrid(x,y)
    xdata = np.vstack((x.ravel(), y.ravel()))
    zdata = z.ravel()

    # Illesztés végrehajtása
    initial_guess = (1,1)  # kezdő tipp az illesztéshez C és aplha
    popt, pcov = curve_fit(f2, xdata, zdata)

    return popt



def f(x,y,C,alpha):
    return C/(x)*np.exp(-alpha*(y)**2/(2*x))
def f2(xy,C,alpha):#az illesztéshez ehhez a formában kell
    x,y = xy
    return C/(x)*np.exp(-alpha*(y)**2/(2*x))
#######
################
###################
# Kód, amely csak közvetlen futtatáskor fut le
if __name__ == "__main__":
    x_grid=np.linspace(1,8,100)
    y_grid= np.linspace(-8,8,100)

    X, Y = np.meshgrid(x_grid, y_grid)

    data = f(X,Y,1,1)
    print(data)
    plt.figure()
    plt.pcolormesh(X, Y, data,cmap='inferno',shading='auto')




    r =np.linspace(1,8,100)
    theta=np.linspace(-np.pi/2,np.pi/2,100)

    #plot2dpolar(1,r,theta)


    #############
    ################
    #############

    DREAM_data = h5py.File('C:\\Users\Csalad\Documents\Gabor\Kutatas\keszthelyi-svn\SOFT\gabor_2023_10\output_Phase3.h5','r')

    f_re =np.array( DREAM_data['eqsys/f_re'][-1,:,:,:])

    p_abs =np.array( DREAM_data['grid/runaway/p1'])

    cosxi = np.array(DREAM_data['grid/runaway/p2'])
    xi = np.arccos(cosxi)

    print(len(f_re[1,:,1]))
    plot2dpolar(np.log(f_re[7,2:,:]),p_abs[:],xi[2:])
    plt.show()

    #######
    #interpoláció:
    ######
    R,Xi =np.meshgrid(p_abs,xi)
    X= R*np.cos(Xi)
    Y= R*np.sin(Xi)
    print('Ide nezz:',X[2:].shape,Y[2:].shape)
    plt.scatter(X[2:],Y[2:])#hogy megnézzük milyen gridre szeretnénk interpoláni
    plt.show()
    #################
    #################
    #uj halo letrehozasa
    #######
    x_uj = p_abs#np.linspace(p_abs[0],p_abs[-1],50)
    y_uj = p_abs#np.linspace(p_abs[0],p_abs[-1]-50,50)

    X_uj,Y_uj =np.meshgrid(x_uj,y_uj)
    

    data=f_re[7,2:,:]
    print("itt vagyok",Y[2:].shape)
    f_gridded = griddata((X[2:].flatten(),Y[2:].flatten()),data.flatten(),(X_uj,Y_uj),method = 'nearest')
    print(Y_uj.shape,f_gridded.shape)
    plot2d(np.log(f_gridded),X_uj,Y_uj)
    plt.show()
    ################################
    ################################
    ###############################
    ### illesztés
    ############################
    # NaN maszk készítése
    mask = ~np.isnan(f_gridded)

    # Érvényes (NaN-mentes) értékek kiválasztása
    x_clean = X_uj[mask]
    y_clean = Y_uj[mask]
    z_clean = f_gridded[mask]


    print(z_clean.ravel().shape,x_clean.ravel().shape,y_clean.ravel().shape)
    param = illeszto(x_clean,y_clean,z_clean)

    print(param)

    ############
    ############
    #illeztett ábrázolása
    ##########



    plt.figure()
    print('Itt van az x_uj->',X_uj.shape)
    plot2d(np.log(f(X_uj,Y_uj,param[0],param[1])),X_uj,Y_uj)
    #plt.pcolormesh(X_uj,Y_uj, np.log(f(X_uj,Y_uj,param[0],param[1])),cmap='inferno',shading='auto')
    plt.show()
    DREAM_data.close()
    """vege"""