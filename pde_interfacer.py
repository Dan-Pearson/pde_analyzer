import matplotlib
matplotlib.use('TkAgg')

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
# implement the default mpl key bindings
from matplotlib.backend_bases import key_press_handler


from matplotlib.figure import Figure

import sys
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk

root = Tk.Tk()
root.wm_title("PDE Interfacer")


def rsym(Qin):
    Qout = np.tril(Qin,-1)+np.tril(Qin,-1).conj().T+np.diag(np.diag(Qin))
    Qout = np.hstack((Qout,np.fliplr(Qout[:,1:-1])))
    Qout = np.vstack((Qout,np.flipud(Qout[1:-1,:])))
    return Qout

def _B21u(N,ksq,dt):
    b21u = np.zeros((int(N/2+1),int(N/2+1)))
    M = 16
    r = np.exp(1j*np.pi*(np.arange(1,int(M+1))-0.5)/M)
    for n in range(int(N/2+1)):
        LR = np.asarray(-dt*np.tile(np.mat(ksq[n:int(N/2+1),n]).T,(1,M)) +\
                        np.tile(r,(int(N/2+1-n),1)))
        b21u[n:,n] = np.real((np.exp(LR/2)-1)/LR).sum(1)/M
    b21u = rsym(b21u)
    return b21u

def _B31u(N,ksq,dt):
    b31u = np.zeros((int(N/2+1),int(N/2+1)))
    M = 16
    r = np.exp(1j*np.pi*(np.arange(1,int(M+1))-0.5)/M)
    for n in range(int(N/2+1)):
        LR = np.asarray(-dt*np.tile(np.mat(ksq[n:int(N/2+1),n]).T,(1,M)) +\
                        np.tile(r,(int(N/2+1-n),1)))
        b31u[n:,n] = np.real(((LR-4)*np.exp(LR/2)+LR+4)/(LR**2)).sum(1)/M
    b31u = rsym(b31u)
    return b31u

def _B32u(N,ksq,dt):
    b32u = np.zeros((int(N/2+1),int(N/2+1)))
    M = 16
    r = np.exp(1j*np.pi*(np.arange(1,int(M+1))-0.5)/M)
    for n in range(int(N/2+1)):
        LR = np.asarray(-dt*np.tile(np.mat(ksq[n:int(N/2+1),n]).T,(1,M)) +\
                        np.tile(r,(int(N/2+1-n),1)))
        b32u[n:,n] = np.real((4*np.exp(LR/2)-2*LR-4)/(LR**2)).sum(1)/M
    b32u = rsym(b32u)
    return b32u

def _B41u(N,ksq,dt):
    b41u = np.zeros((int(N/2+1),int(N/2+1)))
    M = 16
    r = np.exp(1j*np.pi*(np.arange(1,int(M+1))-0.5)/M)
    for n in range(int(N/2+1)):
        LR = np.asarray(-dt*np.tile(np.mat(ksq[n:int(N/2+1),n]).T,(1,M)) +\
                        np.tile(r,(int(N/2+1-n),1)))
        b41u[n:,n] = np.real(((LR-2)*np.exp(LR)+LR+2)/(LR**2)).sum(1)/M
    b41u = rsym(b41u)
    return b41u

def _B43u(N,ksq,dt):
    b43u = np.zeros((int(N/2+1),int(N/2+1)))
    M = 16
    r = np.exp(1j*np.pi*(np.arange(1,int(M+1))-0.5)/M)
    for n in range(int(N/2+1)):
        LR = np.asarray(-dt*np.tile(np.mat(ksq[n:int(N/2+1),n]).T,(1,M)) +\
                        np.tile(r,(int(N/2+1-n),1)))
        b43u[n:,n] = np.real((2*np.exp(LR)-2*LR-2)/(LR**2)).sum(1)/M
    b43u = rsym(b43u)
    return b43u

def _C1u(N,ksq,dt):
    c1u = np.zeros((int(N/2+1),int(N/2+1)))
    M = 16
    r = np.exp(1j*np.pi*(np.arange(1,int(M+1))-0.5)/M)
    for n in range(int(N/2+1)):
        LR = np.asarray(-dt*np.tile(np.mat(ksq[n:int(N/2+1),n]).T,(1,M)) +\
                        np.tile(r,(int(N/2+1-n),1)))
        c1u[n:,n] = np.real((-4-LR+np.exp(LR)*(4-3*LR+LR**2))/LR**3).sum(1)/M
    c1u = rsym(c1u)
    return c1u

def _C23u(N,ksq,dt):
    c23u = np.zeros((int(N/2+1),int(N/2+1)))
    M = 16
    r = np.exp(1j*np.pi*(np.arange(1,int(M+1))-0.5)/M)
    for n in range(int(N/2+1)):
        LR = np.asarray(-dt*np.tile(np.mat(ksq[n:int(N/2+1),n]).T,(1,M)) +\
                        np.tile(r,(int(N/2+1-n),1)))
        c23u[n:,n] = 2*(np.real((2+LR+np.exp(LR)*(-2+LR))/LR**3)).sum(1)/M
    c23u = rsym(c23u)
    return c23u

def _C4u(N,ksq,dt):
    c4u = np.zeros((int(N/2+1),int(N/2+1)))
    M = 16
    r = np.exp(1j*np.pi*(np.arange(1,int(M+1))-0.5)/M)
    for n in range(int(N/2+1)):
        LR = np.asarray(-dt*np.tile(np.mat(ksq[n:int(N/2+1),n]).T,(1,M)) +\
                        np.tile(r,(int(N/2+1-n),1)))
        c4u[n:,n] = np.real((-4-3*LR-LR**2+np.exp(LR)*(4-LR))/LR**3).sum(1)/M
    c4u = rsym(c4u)
    return c4u

#Initial Condition
def initial(N,X,Y):
    u = 0.01*np.random.rand(N,N)
    return u

def _ux(u,dx):
    um1 = np.roll(u,1,axis=1)
    um2 = np.roll(u,2,axis=1)
    up1 = np.roll(u,-1,axis=1)
    up2 = np.roll(u,-2,axis=1)
    return (um2/12 - 2*um1/3 + 2*up1/3 - um2/12)/dx

def _uy(u,dx):
    um1 = np.roll(u,1,axis=0)
    um2 = np.roll(u,2,axis=0)
    up1 = np.roll(u,-1,axis=0)
    up2 = np.roll(u,-2,axis=0)
    return (um2/12 - 2*um1/3 + 2*up1/3 - um2/12)/dx

def rhside(u,dx):
    return _ux(u,dx)**2 + _uy(u,dx)**2

#Exponential time differencing, 4th order Runge-Kutta
#Cox and Matthews
def stepper(u,Eu,Eu2,B21u,B31u,B32u,B41u,B43u,C1u,C23u,C4u,uhat,dx,dt):
    uhat = np.fft.fft2(np.real(u))
    k1u = dt*np.fft.fft2( rhside(np.fft.ifft2(uhat),dx) ) # Nv
    u2hat = Eu2*uhat + B21u*k1u  # a
    k2u = dt*np.fft.fft2( rhside(np.fft.ifft2(u2hat),dx) ) # Na
    u3hat = Eu2*uhat + B31u*k1u + B32u*k2u # b
    k3u = dt*np.fft.fft2( rhside(np.fft.ifft2(u3hat),dx) ) # Nb
    u4hat = Eu*uhat + B41u*k1u + B43u*k3u # c
    k4u = dt*np.fft.fft2( rhside(np.fft.ifft2(u4hat),dx) ) # Nc
    return Eu*uhat + k1u*C1u + (k2u+k3u)*C23u + k4u*C4u # v update, returns uhat

f = Figure(figsize=(5, 4), dpi=100)
a = f.add_subplot(111)

canvas = FigureCanvasTkAgg(f, master=root)
canvas.show()
canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

toolbar = NavigationToolbar2TkAgg(canvas, root)
toolbar.update()
canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)




def on_key_event(event):
    print('you pressed %s' % event.key)
    key_press_handler(event, canvas, toolbar)

canvas.mpl_connect('key_press_event', on_key_event)

e1 = Tk.Entry(root)
e2 = Tk.Entry(root)
e3 = Tk.Entry(root)
Tk.Label(root, text="Linear").pack(side=Tk.LEFT)
e1.pack(side=Tk.LEFT)
Tk.Label(root, text="Nonlinear").pack(side=Tk.LEFT)
e2.pack(side=Tk.LEFT)
Tk.Label(root, text="Nfinal").pack(side=Tk.LEFT)
e3.pack(side=Tk.LEFT)


def _quit():
    root.quit()     # stops mainloop
    root.destroy()  # this is necessary on Windows to prevent
                    # Fatal Python Error: PyEval_RestoreThread: NULL tstate

def isDig(char):
    if char=="0" or char=="1" or char=="2" or char=="3" or char=="4" or char=="5" or char=="6" or char=="7" or char=="8" or char=="9" or char==".":
        return True
    else:
        return False
                    
def convertLinear(myString,kxx,kyy,ksq):
    myString = myString.replace(" ", "")
    linear = (0.+0j)*kxx
    j_prev = 0
    j_next = 0

    for j in range(len(myString)):

        if myString[j] == '+' or myString[j] == '-':
            sign = 1
            if myString[j] == '-':
                sign = -1

            j_next = len(myString)
            for k in range(j+1,len(myString)):
                #print(myString[k])
                if myString[k] == '+' or myString[k] == '-':
                    j_next = k
                    break
                
            myLoop = myString[j_prev+1:j_next]
            #print(j_prev,j_next)
            j_prev = j_next
            j = j_next
            coeff = ""
            nx = 0
            ny = 0
            ns = 0
            nondig_flag = False
            #print(myLoop)
            for m in range(len(myLoop)):
                if myLoop[m] == "\\" or myLoop[m] == "u":
                    nondig_flag = True
                if nondig_flag == False and isDig(myLoop[m]) == True:
                    coeff = coeff + str(myLoop[m])
                if nondig_flag == True and isDig(myLoop[m]) == True:
                    ns = int(int(myLoop[m])/2)
                if myLoop[m] == "x":
                    nx += 1
                if myLoop[m] == "y":
                    ny += 1
            if coeff == "":
                coeff = sign
            coeff = float(coeff)
            
            if ns==0 and nx==0 and ny==0:
                print('error:invalid linear term')
                #print(myLoop)
            elif ns > 0:
                linear += coeff*(-ksq)**ns
            else:
                linear += coeff*(1j*kxx)**nx*(1j*kyy)**ny
                
    return linear

def _refresh():
    #print(e1.get())
    N=128
    Nfinal=int(e3.get())
    dt=0.1
    L=50
    kx = (np.pi/L)*np.hstack((np.arange(0,N/2+1),np.arange(-N/2+1,0)))
    kxx,kyy = np.meshgrid(kx,kx)
    ksq = kxx**2 + kyy**2
    LL = convertLinear(e1.get(),kxx,kyy,ksq)
    
    def rhside(u,dx):
        return _ux(u,dx)**2 + _uy(u,dx)**2

    B21u = _B21u(N,ksq,dt)
    B31u = _B31u(N,ksq,dt)
    B32u = _B32u(N,ksq,dt)
    B41u = _B41u(N,ksq,dt)
    B43u = _B43u(N,ksq,dt)
    C1u = _C1u(N,ksq,dt)
    C23u = _C23u(N,ksq,dt)
    C4u = _C4u(N,ksq,dt)

    x = (2*L/N)*np.arange(-N//2,N//2)
    dx = float(np.abs(x[1]-x[0]))
    X,Y = np.meshgrid(x, x) # used for initial condition
    u_init = initial(N,X,Y)
    u = 1.*u_init
    Eu = np.exp(dt*LL)
    Eu2 = np.exp(dt*LL/2.)
    del X # free up unneeded memory
    del Y #
    uhat = np.fft.fft2(u)
    #ETD Runge-Kutta
    for n in range(Nfinal+1):
        uhat = stepper(u,Eu,Eu2,B21u,B31u,B32u,B41u,B43u,C1u,C23u,C4u,uhat,dx,dt)
        u = np.real(np.fft.ifft2(uhat))
    a.clear()
    a.imshow(u)
    canvas.draw()
        
button = Tk.Button(master=root, text='Refresh', command=_refresh)
button.pack(side=Tk.BOTTOM)

button = Tk.Button(master=root, text='Quit', command=_quit)
button.pack(side=Tk.BOTTOM)


Tk.mainloop()
