import os
#import sys
import glob
#import cPickle as pickle
import numpy as np
#import matplotlib.pyplot as plt
filter="*.swc"  
all_f = glob.glob(filter)
#f= all_f[0]
for f in all_f:
#        fl = []
#        fl = np.loadtxt(f, dtype=float, comments="#")
        with open(f) as flr:
            fls = flr.read().splitlines()
        i=0
        fl=np.empty((len(fls),7),float)
        for fll in fls:
            X = fls[i].split()
            X = np.array([X])
            X = X.astype(float)
            if len(X[0,:])==6:
                X=np.append([X], [-1])
            fl[i,:]= X 
            i=i+1
        
        sNeu = np.empty((0,7),float)
        Px= np.where(fl[:,6]==-1)
        Px=list(Px[0])
        while len(Px)>0:
            P = Px[0]
            Px = Px[1:]
            while P.size>0:
                P=int(P)
                sNeu = np.vstack((sNeu,fl[P,:]))
                Child= np.where(fl[:,6]==fl[P,0])
                Child = list(Child[0]) 
                if len(Child)==0:
                    break
                if len(Child)>1:
                    Px= np.append(Child[1:],Px)                
                P = Child[0]
        
        sRe = sNeu[:,6]
        Li = list(range(1,(len(sNeu[:,1])+1)))
        Li1 = Li[:-1]
        for i in Li1:
            if sNeu[i,6] != -1:
                pids= np.where(sNeu[:,0]==sNeu[i,6])
                pids = float(pids[0])
                sRe[i] = pids+1
        sNeu[:,6] = sRe
        sNeu[:,0]= Li
        if not os.path.exists("out"):
            os.mkdir("out")
        os.chdir(".\out")      
        np.savetxt(f,sNeu,fmt="%u %u %f %f %f %f %d")
        os.chdir("..")

