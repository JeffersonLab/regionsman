# Useful functions for working with SIDIS
import numpy as np
sq2 = np.sqrt(2)


# Nachtmann x
def xnac(xb,Mpp,QQ):
    val = 2*xb/(1+np.sqrt(1+4*xb**2*Mpp**2/(float(QQ)**2)))
 
    return val

# Nachtmann z
def zn(zh,xb,QQ,Mpp,Mhh,pt):
    term1 = xnac(xb,Mpp,QQ)*zh/xb/2.0
    term2 =np.sqrt(Mhh**2+pt**2)
    term3 = 1+np.sqrt(1-(4*Mpp**2)*term2*xb**2/(zh**2*QQ**4))
    return term1*term3

#Nactmann z in terms of qt
def znqt(zh,xb,Q,Mpp,Mhh,qt):
    Qf = Q**4.0
    Mpsq = Mpp**2.0
    Mhsq = Mhh**2.0
    xsq = xb**2.0
    zsq = zh**2.0
    xval = xnac(xb,Mpp,Q)
    xnsq = xval**2.0
    term1 = Qf*xval*zh/(2*xb*(Qf+(xnsq*Mpsq*qt**2)))
    term2 = (4*Mpsq*Mhsq*xsq*(Qf+(xnsq*Mpsq*qt**2)))/(Qf**2*zsq)
    term3 = np.sqrt(1-term2)
    term4 = 1+term3
    term5 = term1*term4
    return term5


### Light Cone Products

def lc(a,b):
    val = (a[0]*b[1])+(a[1]*b[0])-(a[2]*b[2])-(a[3]*b[3])
    return val
def TwoDprod(a,b):
    val = a[0]*b[0]+a[1]*b[1]
    return val

# Setting Up Four-Vectors

def qvec(Q):
    return np.array([-Q/sq2,Q/sq2,0,0])

def Phvec(Mhad, qt, z,Q):
    val = np.array([((Mhad**2)+(z**2)*(qt**2))/(z*Q*np.sqrt(2)),(z*Q)/(np.sqrt(2)),-qt*z,0])
    return val

def kinitvec(Q,xhat,kinit):
    val = np.array([(Q/(xhat*np.sqrt(2))),-((kinit**2)*(xhat))/(Q*np.sqrt(2)),0,0])
    return val

# Two Vectors and Kfinal

def TwoVecPh(zhat,qt): # Transverse part of hadron P_b
    val = np.array([zhat*qt,0])
    return val

def TwoVecIntr(deltk,ang): # Intrinsic transverse vector
    val = np.array([deltk*np.cos(ang),deltk*np.sin(ang)])
    return val

def TwoVecFin(deltk,qt,zhat,ang): # Total transverse vector for kf
    val = TwoVecIntr(deltk,ang)-TwoVecPh(zhat,qt)
    return val

def kTwoDsq(deltk,qt,zhat,ang): # k_(f,T)^2
    val = TwoDprod(TwoVecFin(deltk,qt,zhat,ang),TwoVecFin(deltk,qt,zhat,ang))
    return val

def kfinvec(Q,zhat,kfin,deltk,ang,qt): # k_f, kfin^2 = kf^2
    val = np.array([((kfin**2)+kTwoDsq(deltk,qt,zhat,ang))/(Q*zhat*np.sqrt(2)),(zhat*Q)/(np.sqrt(2))\
                    ,deltk*np.cos(ang)-qt*zhat,deltk*np.sin(ang)])
    return val

# More Four Vector Definitions

def kFourvec(Q,zhat,kfinal,kt,ang,qt): # k
    return kfinvec(Q,zhat,kfinal,kt,ang,qt)-qvec(Q)

def kFourvecsq(Q,zhat,kfinal,kt,ang,qt): # k^2
    return lc(kFourvec(Q,zhat,kfinal,kt,ang,qt),kFourvec(Q,zhat,kfinal,kt,ang,qt))

def kx(Q,xhat,zhat,kfinal,kt,ang,qt,kinit):
    val = -kfinvec(Q,zhat,kfinal,kt,ang,qt)+qvec(Q)+kinitvec(Q,xhat,kinit)
    return val

def kxsqr(Q,xhat,zhat,kfinal,kt,ang,qt,kinit):
    val = lc(kx(Q,xhat,zhat,kfinal,kt,ang,qt,kinit),kx(Q,xhat,zhat,kfinal,kt,ang,qt,kinit))
    return val

# Region Ratios

def R0(kfinal,kinit,deltk,QQ): # Overall Hardness
    val = (np.max([kfinal**2,kinit**2,deltk**2]))/(1.0*QQ**2)
    return val

def R1(Q,xhat,zhat,kfin,kinit,deltk,ang,qt,Mhad,z):
    val = lc(Phvec(Mhad,qt,z,Q),kfinvec(Q,zhat,kfin,deltk,ang,qt))/lc(Phvec(Mhad,qt,z,Q)\
    ,kinitvec(Q,xhat,kinit))
    return val

def R2(QQ,zhat,kfinal,kt,ang,qt): # Transverse Hardness
    val = (np.abs(kFourvecsq(QQ,zhat,kfinal,kt,ang,qt)))/(float(QQ**2))                         
    return val

def R3(Q,xhat,zhat,kfinal,kt,ang,qt,kinit): # Higher Order Importance.
    val = np.abs(kxsqr(Q,xhat,zhat,kfinal,kt,ang,qt,kinit))/(float(Q**2))
    return val

# Zhatv1 in terms of kx^2

def zhatv1(Q,kinit,kfinal,ang,qt,xhat,kt,kx):
    isq = kinit**2.0
    fsq = kfinal**2.0
    QQ = Q**2.0
    Qf = Q**4.0
    qsq = qt**2.0
    xsq = xhat**2.0
    ktsq = kt**2.0
    kxsq = kx**2.0
    
    val = -((Qf +fsq*QQ*xhat-isq*QQ*xhat -kxsq*QQ*xhat-Qf*xhat+isq*QQ*xsq\
          +2*kt*QQ*qt*xhat*np.cos(ang)-2*isq*kt*qt*xsq*np.cos(ang)+\
           np.sqrt(4*(fsq+ktsq)*xhat*(QQ-isq*xhat)*(Qf*(-1+xhat)-QQ*qsq*xhat\
           +isq*qsq*xsq)+(Qf*(-1+xhat)+QQ*xhat*(-fsq+isq+kxsq-isq*xhat)+\
            2*kt*qt*xhat*(-QQ+isq*xhat)*np.cos(ang))**2))\
           /(2*(Qf*(-1+xhat)-QQ*qsq*xhat+isq*qsq*xsq)))
    return val

