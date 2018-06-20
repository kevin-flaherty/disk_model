#Eccentric debris disk modelling code
#Based on code by Evan Carter and Kevin Flaherty
#Edited once again by David Vizgan (aka Mr. David)

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy import ndimage
from astropy import constants as const

#testing time
import time

class Disk:
    'Common class for circumstellar disk structure'
    
    #Define useful constants
    
    AU = const.au.cgs.value          # - astronomical unit (cm)
    Rsun = const.R_sun.cgs.value     # - radius of the sun (cm)
    c = const.c.cgs.value            # - speed of light (cm/s)
    h = const.h.cgs.value            # - Planck's constant (erg/s)
    kB = const.k_B.cgs.value         # - Boltzmann's constant (erg/K)
    pc = const.pc.cgs.value          # - parsec (cm)
    sigmaB  = const.sigma_sb.cgs.value # - Stefan-Boltzmann constant (erg cm^-2 s^-1 K^-4) #only difference in defined constants 
    Jy = 1.e23                       # - cgs flux density (Janskys)
    Lsun = const.L_sun.cgs.value     # - luminosity of the sun (ergs)
    Mearth = const.M_earth.cgs.value # - mass of the earth (g)
    mh = const.m_p.cgs.value         # - proton mass (g)
    Da = mh                          # - atomic mass unit (g)
    Msun = const.M_sun.cgs.value     # - solar mass (g)
    G = const.G.cgs.value            # - gravitational constant (cm^3/g/s^2)
    rad = 206264.806                 # - radian to arcsecond conversion
    kms = 1e5                        # - convert km/s to cm/s
    GHz = 1e9                        # - convert from GHz to Hz
    mCO = 12.011+15.999              # - CO molecular weight
    mHCO = mCO+1.008-0.0005          # - HCO molecular weight
    mu = 2.37                        # - gas mean molecular weight
    m0 = mu*mh                       # - gas mean molecular opacity
    Hnuctog = 0.706*mu               # - H nuclei abundance fraction (H nuclei:gas)
    sc = 1.59e21                     # - Av --> H column density (C. Qi 08,11)
    H2tog = 0.8                      # - H2 abundance fraction (H2:gas)
    Tco = 19.                        # - freeze out
    sigphot = 0.79*sc                # - photo-dissociation column
    
    def __init__(self,params=[-0.5,0.09,1.,10.,1000.,150.,51.5,2.3,1e-4,0.01,33.9,19.,69.3,-1,0,0,[.76,1000],[10,800],0.09,0.1],obs=[180,131,300,170],rtg=True,vcs=True,line='co',sh_relation='linear',ring=None):

        tb = time.clock()
        self.ring=ring
        self.set_obs(obs)   # set the observational parameters
        self.set_params(params) # set the structure parameters

        self.set_structure(sh_relation)  # use obs and params to create disk structure
        if rtg:
            self.set_rt_grid()
            self.set_line(line=line,vcs=vcs)
            
        'need to set a parameter for scale height here'

        tf = time.clock()
        print "disk init took {t} seconds".format(t=(tf-tb))

    def set_params(self,params):
        'Set the disk structure parameters'
        self.qq = params[0]                 # - temperature index
        self.Mdust = params[1]*Disk.Msun    # - dust mass
        self.pp = params[2]                 # - surface density index
        self.Ain = params[3]*Disk.AU        # - inner edge in cm
        self.Aout = params[4]*Disk.AU       # - outer edge in cm
        self.Rc = params[5]*Disk.AU         # - critical radius in cm
        self.thet = math.radians(params[6]) # - convert inclination to radians
        self.Mstar = params[7]*Disk.Msun    # - convert mass of star to g
        self.Xco = params[8]                # - CO gas fraction
        self.vturb = params[9]*Disk.kms     # - turbulence velocity
        self.zq0 = params[10]               # - Zq, in AU, at 150 AU
        self.tmid0 = params[11]             # - Tmid at 150 AU
        self.tatm0 = params[12]             # - Tatm at 150 AU
        self.handed = params[13]            # 
        self.ecc = params[14]               # - eccentricity of disk
        self.aop = math.radians(params[15]) # - angle between los and perapsis convert to radians
        self.sigbound = [params[16][0]*Disk.sc,params[16][1]*Disk.sc] #-upper and lower column density boundaries
        if len(params[17])==2:
            # - inner and outer abundance boundaries
            self.Rabund = [params[17][0]*Disk.AU,params[17][1]*Disk.AU]
        else:
            self.Rabund=[params[17][0]*Disk.AU,params[17][1]*Disk.AU,params[17][2]*Disk.AU,params[17][3]*Disk.AU,params[17][4]*Disk.AU,params[17][5]*Disk.AU]
        self.costhet = np.cos(self.thet)  # - cos(i)
        self.sinthet = np.sin(self.thet)  # - sin(i)
        self.cosaop = np.cos(self.aop)
        self.sinaop = np.sin(self.aop)
        if self.ring is not None:
            self.Rring = self.ring[0]*Disk.AU # location of ring
            self.Wring = self.ring[1]*Disk.AU # width of ring
            self.sig_enhance = self.ring[2] # surface density enhancement (a multiplicative factor) above the background
        self.Lstar      = params[18]
        self.sh_param   = params[19]
        self.kap = 2.3
        
    def set_obs(self,obs):
        'Set the observational parameters. These parameters are the number of r, phi, S grid points in the radiative transer grid, along with the maximum height of the grid.'
        self.nr = obs[0]
        self.nphi = obs[1]
        self.nz = obs[2]
        self.zmax = obs[3]*Disk.AU
           
        
    'Main edits to the code go here, David.'

    def set_structure(self, sh_relation):
        #tst=time.clock()
        '''Calculate the disk density and temperature structure given the specified parameters'''
        # Define the desired regular cylindrical (r,z) grid
        nac = 500#256             # - number of unique a rings
        #nrc = 256             # - numver of unique r points
        amin = self.Ain       # - minimum a [AU]
        amax = self.Aout      # - maximum a [AU]
        e = self.ecc          # - eccentricity
        nzc = int(2.5*nac)#nac*5           # - number of unique z points
        zmin = .1*Disk.AU      # - minimum z [AU]
        nfc = self.nphi       # - number of unique f points
        af = np.logspace(np.log10(amin),np.log10(amax),nac)
        zf = np.logspace(np.log10(zmin),np.log10(self.zmax),nzc)
        pf = np.linspace(0,2*np.pi,self.nphi) #f is with refrence to semi major axis
        ff = (pf - self.aop) % (2*np.pi) # phi values are offset by aop- refrence to sky
        rf = np.zeros((nac,nfc))
        for i in range(nac):
            for j in range(nfc):
                rf[i,j] = (af[i]*(1.-e*e))/(1.+e*np.cos(ff[j]))

        idz = np.ones(nzc)
        idf = np.ones(self.nphi)
        #rcf = np.outer(rf,idz)
        ida = np.ones(nac)
        ##zcf = np.outer(ida,zf)
        ##acf = af[:,np.newaxis]*np.ones(nzc)
        #order of dimensions: a, f, z
        pcf,acf,zcf = np.meshgrid(pf,af,zf)
        #zcf = (np.outer(ida,idf))[:,:,np.newaxis]*zf
        #pcf = (np.outer(ida,pf))[:,:,np.newaxis]*idz
        fcf = (pcf - self.aop) % (2*np.pi)
        #acf = (np.outer(af,idf))[:,:,np.newaxis]*idz
        rcf=rf[:,:,np.newaxis]*idz
        #print "coords init {t}".format(t=time.clock()-tst)
        
        if 0:
            print 'plotting'
            plt.plot((rcf*np.cos(fcf)).flatten(),(rcf*np.sin(fcf)).flatten())
            plt.show()

        # rcf[0][:] = radius at all z for first radial bin
        # zcf[0][:] = z in first radial bin

        # Here introduce new z-grid (for now just leave old one in)

        # Interpolate dust temperature and density onto cylindrical grid
        ###### doesnt seem to be used anywhere ######
        #tf = 0.5*np.pi-np.arctan(zcf/rcf)  # theta values
        #rrf = np.sqrt(rcf**2.+zcf**2)

        # bundle the grid for helper functions
        ###### add angle to grid? ######
        grid = {'nac':nac,'nfc':nfc,'nzc':nzc,'rcf':rcf,'amax':amax,'zcf':zcf}#'ff':ff,'af':af,
        self.grid=grid

        #print "grid {t}".format(t=time.clock()-tst)
        #define temperature structure
        # use Dartois (03) type II temperature structure
        ###### expanding to 3D should not affect this ###### 
        delta = 1.                # shape parameter
        rcf150=rcf/(150.*Disk.AU)
        rcf150q=rcf150**self.qq
        zq = self.zq0*Disk.AU*rcf150**1.3
        #zq = self.zq0*Disk.AU*(rcf/(150*Disk.AU))**1.1
        tmid = self.tmid0*rcf150q
        tatm = self.tatm0*rcf150q
        tempg = (self.Lstar * self.Lsun / (16. * np.pi * rcf**2 * self.sigmaB))**0.25
        ii = zcf > zq
        tempg[ii] = tatm[ii]
        #Type I structure
#        tempg = tmid*np.exp(np.log(tatm/tmid)*zcf/zq)
        ###### this step is slow!!! ###### 
        #print "temp struct {t}".format(t=time.clock()-tst)
        
        "Lines 183 - 207 are where you change the density structure. This needs to be changed from gas to dust."
        
        # Calculate vertical density structure
        # nolonger use exponential tail
        ## Circular:
        #Sc = self.McoG*(2.-self.pp)/(2*np.pi*self.Rc*self.Rc)
        #siggas = Sc*(rf/self.Rc)**(-1*self.pp)*np.exp(-1*(rf/self.Rc)**(2-self.pp))
        ## Elliptical:
        #asum = (np.power(af,-1*self.pp)).sum()
        rp1 = np.roll(rf,-1,axis=0)
        rm1 = np.roll(rf,1,axis=0)
        #*** Approximations used here ***#
        #siggas = (self.McoG*np.sqrt(1.-e*e))/((rp1-rm1)*np.pi*(1.+e*np.cos(fcf[:,:,0]))*np.power(acf[:,:,0],self.pp+1.)*asum)
        #siggas[0,:] = (self.McoG*np.sqrt(1.-e*e))/((rf[1,:]-rf[0,:])*2.*np.pi*(1.+e*np.cos(ff))*np.power(af[0]*idf,self.pp+1.)*asum)
        #siggas[nac-1,:] = (self.McoG*np.sqrt(1.-e*e))/((rf[nac-1,:]-rf[nac-2,:])*2.*np.pi*(1.+e*np.cos(ff))*np.power(af[nac-1]*idf,self.pp+1.)*asum)
        Sc = self.Mdust * (self.pp + 2.) / (2. * np.pi * (self.Rout**(2. + self.pp) - self.Rin**(2. + self.pp))) 
        siggas_r = Sc * (rcf ** self.pp)
        #Sc = self.McoG*(2.-self.pp)/((amax**(2-self.pp)-amin**(2-self.pp)))
        #siggas_r = Sc*acf[:,:,0]**(-1*self.pp)
        dsdth = (acf[:,:,0]*(1-e*e)*np.sqrt(1+2*e*np.cos(fcf[:,:,0])+e*e))/(1+e*np.cos(fcf[:,:,0]))**2
        siggas = (siggas_r/(self.sh_param*np.sqrt(np.pi)))*(e**(-1*(zf/self.sh_param)))

        ## Add an extra ring
        if self.ring is not None:
            w = np.abs(rcf-self.Rring)<self.Wring/2.
            if w.sum()>0:
                tempg[w] = tempg[w]*(rcdf[w]/(150*Disk.AU))**(self.sig_enhance-self.qq)/((rcf[w].max())/(150.*Disk.AU))**(-self.qq+self.sig_enhance)

        ##calculate the scale height of the disk
        self.set_scale_height(sh_relation, rcf)
                
        #print "surface density {t}".format(t=time.clock()-tst)
        if 0:
            print 'plotting'
            #plt.pcolor(rcf[:,:,0]*np.cos(fcf[:,:,0]),rcf[:,:,0]*np.sin(fcf[:,:,0]),(siggas[:,:]))
            plt.loglog(rcf[:,0,0]/self.AU,siggas[:,0],color='k',lw=2)
            plt.loglog(rcf[:,nfc/2,0]/self.AU,siggas[:,nfc/2],color='r',lw=2)
            plt.loglog(rcf[:,0,0]/self.AU,linrho[:,0],ls='--',lw=2,color='k')
#            plt.loglog(rcf[:,0,0]/self.AU,siggas_r[:,0],ls=':',lw=2,color='k')
            plt.loglog(rcf[:,nfc/2,0]/self.AU,linrho[:,nfc/2],ls='--',lw=2,color='r')
#            plt.loglog(rcf[:,nfc/2,0]/self.AU,siggas_r[:,nfc/2],ls=':',lw=2,color='r')
#            plt.colorbar()
            plt.show()
            
        if 0:
            # check that siggas adds up to Mdisk #
            df=ff[1]-ff[0]
            dA = 0.5*(rp1-rm1)*df*rf
            dA[0,:]=(rf[1,:]-rf[0,:])*rf[0,:]*df
            dA[nac-1,:]=(rf[nac-1,:]-rf[nac-2,:])*rf[nac-1,:]*df
            mcheck=(siggas*dA)
            mcheck=mcheck.sum()
            #print "sig mass check (should be 1)"
            #print mcheck/self.McoG

            #dsdth = (acf*(1-e*e)*np.sqrt(1+2*e*np.cos(fcf)+e*e))/(1+e*np.cos(fcf))**2
            dr = af-np.roll(af,1)
            dr[0] = af[0]
            dr = dr[:,np.newaxis]*np.ones(nfc)
            dm = (siggas*dr*acf[:,:,0]*df)
            #dm = (linrho*dA*dsdth*2*np.pi)
#dm[0] = 0
            print 'second sig mass check ',dm.sum()/self.McoG
                  

        self.calc_hydrostatic(tempg,siggas,grid)
        
        if 0:
            #check if rho0 adds up to Mdisk
            df=2*np.pi/self.nphi
            #dz=0.5*(np.roll(zcf,-1,axis=2)-np.roll(zcf,1,axis=2))
            #dz[:,:,0]=zcf[:,:,1]-zcf[:,:,0]
            #dz[:,:,nzc-1]=zcf[:,:,nzc-1]-zcf[:,:,nzc-2]
            #dr=0.5*(np.roll(rcf,-1,axis=0)-np.roll(rcf,1,axis=0))
            #dr[0]=rcf[1]-rcf[0]
            #dr[nac-1]=rcf[nac-1]-rcf[nac-2]
            dz = zcf-np.roll(zcf,1,axis=2)
            dz[:,:,0] = 0#zcf[:,:,1]
            dr = acf-np.roll(acf,1,axis=0)
            dr[0] = 0#rcf[1]
            dV=acf*df*dr*dz
            mcheck=self.rho0*dV
            mcheck=mcheck.sum()
            print "rho mass check (should be 1/2 as z is only one half of disk)"
            print mcheck/self.McoG

        #print "hydro done {t}".format(t=time.clock()-tst)
        #Calculate radial pressure differential
        ### nolonger use pressure term ###
        #Pgas = Disk.kB/Disk.m0*self.rho0*tempg
        #dPdr = (np.roll(Pgas,-1,axis=0)-Pgas)/(np.roll(rcf,-1,axis=0)-rcf)
        #print dPdr[:5,0,0],dPdr[200:205,0,500]
        #dPdr = 0#(np.roll(Pgas,-1,axis=0)-Pgas)/(np.roll(rcf,-1,axis=0)-rcf)
        
        
        #Calculate velocity field
        #Omg = np.sqrt((dPdr/(rcf*self.rho0)+Disk.G*self.Mstar/(rcf**2+zcf**2)**1.5))
        #w = np.isnan(Omg)
        #if w.sum()>0:
        #    Omg[w] = np.sqrt((Disk.G*self.Mstar/(rcf[w]**2+zcf[w]**2)**1.5))
        
        #https://pdfs.semanticscholar.org/75d1/c8533025d0a7c42d64a7fef87b0d96aba47e.pdf
        #Lovis & Fischer 2010, Exoplanets edited by S. Seager (eq 11 assuming m2>>m1)
        self.vel = np.sqrt(Disk.G*self.Mstar/(acf*(1-self.ecc**2.)))*(np.cos(self.aop+fcf)+self.ecc*self.cosaop)

        ###### Major change: vel is linear not angular ######
        #Omk = np.sqrt(Disk.G*self.Mstar/acf**3.)#/rcf
        #velrot = np.zeros((3,nac,nfc,nzc)) 
        #x,y velocities with refrence to semimajor axis (f)
        #velx = (-1.*Omk*acf*np.sin(fcf))/np.sqrt(1.-self.ecc**2)
        #vely = (Omk*acf*(self.ecc+np.cos(fcf)))/np.sqrt(1.-self.ecc**2)
        #x,y velocities with refrence to sky (phi) only care about Vy on sky
        #velrot[0] = self.cosaop*vel[0] - self.sinaop*vel[1]
        #velrot = self.sinaop*velx + self.cosaop*vely

        #velrot = np.sqrt((Disk.G*self.Mstar)/(acf*(1.-self.ecc**2)))*(self.sinaop*(-1.*np.sin(fcf)) + self.cosaop*(self.ecc+np.cos(fcf)))
        #velrot2 = np.sqrt((Disk.G*self.Mstar)*(2/rcf-1/acf))

        if 0:
            plt.subplot(211)
            #plt.plot(ff/np.pi,velrot[nac/2,:,0]/1e5,'.',color='k',lw=2)
            plt.plot(pf/np.pi,(Omg*rcf)[nac/2,:,0]/1e5,'.',lw=2)
            #plt.plot(ff/np.pi,velrot2[nac/2,:,0]/1e5,'.',color='r')
            plt.subplot(212)
            #plt.plot(af/Disk.AU,velrot[:,0,0]/1e5,color='k',lw=2)
            plt.plot(af/Disk.AU,(Omg*rcf)[:,0,0]/1e5,lw=2)
            #plt.plot(af/Disk.AU,velrot2[:,0,0]/1e5,color='r')
            #plt.plot(af/Disk.AU,velrot[:,nfc/2,0]/1e5,color='k',ls='--')
            plt.plot(af/Disk.AU,(Omg*rcf)[:,nfc/2,0]/1e5,ls='--',lw=2)
            #plt.plot(af/Disk.AU,velrot2[:,nfc/2,0]/1e5,color='r',ls='--')


        #print "Vel {t}".format(t=time.clock()-tst)
        if 0:
            print 'plotting velocity'
            plt.clf()
            '''
            plt.plot(fcf[0,:,0],velrot[0,0,:,0],label='Vx')
            plt.plot(fcf[0,:,0],velrot[1,0,:,0],label='Vy')
            plt.plot(fcf[0,:,0],np.sqrt(Disk.G*self.Mstar*((2./rcf[0,:,0])-(1./acf[0,:,0]))),"o",color="c",label="Vis Viva")
            plt.plot(fcf[0,:,0],np.sqrt(velrot[0,0,:,0]**2+velrot[1,0,:,0]**2),"x",color="k",label='V')
            plt.legend(loc = "lower right")
            plt.show()
            '''
            plt.subplot(131,aspect="equal")
            #plt.axes().set_aspect("equal")
            plt.title("Vx/V")
            plt.pcolor(rcf[:,:,0]*np.cos(pcf[:,:,0]),rcf[:,:,0]*np.sin(pcf[:,:,0]),velrot[0,:,:,0])#/np.sqrt(velrot[0,:,:,0]**2+velrot[1,:,:,0]**2))
            plt.colorbar()
            plt.subplot(132,aspect="equal")
            #plt.axes().set_aspect("equal")
            plt.title("Vy/V")
            plt.pcolor(rcf[:,:,0]*np.cos(pcf[:,:,0]),rcf[:,:,0]*np.sin(pcf[:,:,0]),velrot[1,:,:,0])#/np.sqrt(velrot[0,:,:,0]**2+velrot[1,:,:,0]**2))
            plt.colorbar()
            plt.subplot(133,aspect="equal")
            #plt.axes().set_aspect("equal")
            plt.title("Log V")
            plt.pcolor(rcf[:,:,0]*np.cos(pcf[:,:,0]),rcf[:,:,0]*np.sin(pcf[:,:,0]),np.log10(np.sqrt(velrot[0,:,:,0]**2+velrot[1,:,:,0]**2)))
            plt.colorbar()
            plt.show()
        
        # Check for NANs
        ### nolonger use Omg ###
        #ii = np.isnan(Omg)
        #Omg[ii] = Omk[ii]
        ii = np.isnan(self.rho0)
        if ii.sum() > 0:
            self.rho0[ii] = 1e-60
            print 'Beware: removed NaNs from density (#%s)' % ii.sum()
        ii = np.isnan(tempg)
        if ii.sum() > 0:
            tempg[ii] = 2.73
            print 'Beware: removed NaNs from temperature (#%s)' % ii.sum()

        #print "nan chekc {t}".format(t=time.clock()-tst)
        # find photodissociation boundary layer from top
        zpht_up = np.zeros((nac,nfc))
        zpht_low = np.zeros((nac,nfc))
        sig_col = np.zeros((nac,nfc,nzc))
        #zice = np.zeros((nac,nfc))
        for ia in range(nac):
            for jf in range (nfc):
                psl = (Disk.Hnuctog/Disk.m0*self.rho0[ia,jf,:])[::-1]
                zsl = self.zmax - (zcf[ia,jf,:])[::-1]
                foo = (zsl-np.roll(zsl,1))*(psl+np.roll(psl,1))/2.
                foo[0] = 0
                nsl = foo.cumsum()
                sig_col[ia,jf,:] = nsl[::-1]*Disk.m0/Disk.Hnuctog
                pht = (np.abs(nsl) >= self.sigbound[0])
                if pht.sum() == 0:
                    zpht_up[ia,jf] = np.min(self.zmax-zsl)
                else:
                    zpht_up[ia,jf] = np.max(self.zmax-zsl[pht])
                #Height of lower column density boundary
                pht = (np.abs(nsl) >= self.sigbound[1])
                if pht.sum() == 0:
                    zpht_low[ia,jf] = np.min(self.zmax-zsl)
                else:
                    zpht_low[ia,jf] = np.max(self.zmax-zsl[pht])
                #used to be a seperate loop
                ###### only used for plotting
                #foo = (tempg[ia,jf,:] < Disk.Tco)
                #if foo.sum() > 0:
                #    zice[ia,jf] = np.max(zcf[ia,jf,foo])
                #else:
                #    zice[ia,jf] = zmin
        self.sig_col = sig_col
        #szpht = zpht
        #print "Zpht {t} seconds".format(t=(time.clock()-tst))
                    
        '''
        
        szpht = zpht
        #zpht = scipy.signal.medfilt(zpht,kernel_size=7) #smooth it

        # find height where CO freezes out
        # only used for ploting
        zice = np.zeros(nrc)
        for ir in range(nrc):
            foo = (tempg[ir,:] < Disk.Tco)
            if foo.sum() > 0:
                zice[ir] = np.max(zcf[ir,foo])
            else:
                zice[ir] = zmin
        '''
        self.af = af
        #self.ff = ff
        #self.rf = rf
        self.pf = pf
        self.nac = nac
        self.zf = zf
        self.nzc = nzc
        self.tempg = tempg
        #self.Omg0 = Omg#velrot
        self.zpht_up = zpht_up
        self.zpht_low = zpht_low
        self.pcf = pcf  #only used for plotting can remove after testing
        self.rcf = rcf  #only used for plotting can remove after testing
        
        
        
        
        if 0:
            plt.figure()
            cs = plt.contour(rcf/Disk.AU,zcf/Disk.AU,np.log10(self.rho0/(Disk.mu*Disk.mh)),np.arange(0,10,1))
            #cs2 = plt.contour(tr[0,:,:]/Disk.AU,tdiskZ[0,:,:]/Disk.AU,np.log10(self.rhoG[0,:,:])+4,np.arange(0,11,1))
            cs2 = plt.contour(rcf/Disk.AU,zcf/Disk.AU,tempg,(30,50,70,100,120,150),colors='k')
            #cs3 = plt.contour(tr[0,:,:]/Disk.AU,tdiskZ[0,:,:]/Disk.AU,tT[0,:,:],(20,40,60,80,100,120),colors='k',ls='--')
            #plt.plot(tr[0,:,:]/Disk.AU,zpht[0,:,:]/Disk.AU,color='k',lw=8,ls='--')
            plt.plot(rf/Disk.AU,zice/Disk.AU,color='b',lw=6,ls='--')
            plt.plot(rf/Disk.AU,szpht/Disk.AU,color='k',lw=6,ls='--')
            plt.colorbar(cs,label='log n')
            plt.clabel(cs2,fmt='%1i')
            #plt.clabel(cs3,fmt='%3i')
            plt.xlim(0,500)
            plt.xlabel('R (AU)',fontsize=20)
            plt.ylabel('Z (AU)',fontsize=20)
            plt.show()

    def set_rt_grid(self):
        #tst=time.clock()
        ### Start of Radiative Transfer portion of the code...
        # Define and initialize cylindrical grid
        Smin = 1*Disk.AU                 # offset from zero to log scale
        if self.thet > np.arctan(self.Aout/self.zmax):
            Smax = 2*self.Aout/self.sinthet
        else:
            Smax = 2.*self.zmax/self.costhet       # los distance through disk
        Smid = Smax/2.                    # halfway along los
        ytop = Smax*self.sinthet/2.       # y origin offset for observer xy center
        #sky coordinates
        #R = np.logspace(np.log10(self.Ain*(1-self.ecc)),np.log10(self.Aout*(1+self.ecc)),self.nr)
        R = np.linspace(0,self.Aout*(1+self.ecc),self.nr) #******* not on cluster*** #
        phi = np.arange(self.nphi)*2*np.pi/(self.nphi-1)
        foo = np.floor(self.nz/2)
        
        #S_old = np.concatenate([Smid+Smin-10**(np.log10(Smid)+np.log10(Smin/Smid)*np.arange(foo)/(foo)),Smid-Smin+10**(np.log10(Smin)+np.log10(Smid/Smin)*np.arange(foo)/(foo))]) 
        S_old = np.arange(2*foo)/(2*foo)*(Smax-Smin)+Smin #*** not on cluster**
        
        #print "grid {t}".format(t=time.clock()-tst)
        # Basically copy S_old, with length nz,  into each column of a nphi*nr*nz matrix
        S = (S_old[:,np.newaxis,np.newaxis]*np.ones((self.nr,self.nphi))).T

        # arrays in [phi,r,s] on sky coordinates
        X = (np.outer(R,np.cos(phi))).transpose()
        Y = (np.outer(R,np.sin(phi))).transpose()

        # transform grid to disk coordinates
        tdiskZ = self.zmax*(np.ones((self.nphi,self.nr,self.nz)))-self.costhet*S
        if self.thet > np.arctan(self.Aout/self.zmax):
            tdiskZ -=(Y*self.sinthet).repeat(self.nz).reshape(self.nphi,self.nr,self.nz)
        tdiskY = ytop - self.sinthet*S + (Y/self.costhet).repeat(self.nz).reshape(self.nphi,self.nr,self.nz)
        tr = np.sqrt(X.repeat(self.nz).reshape(self.nphi,self.nr,self.nz)**2+tdiskY**2)
        tphi = np.arctan2(tdiskY,X.repeat(self.nz).reshape(self.nphi,self.nr,self.nz))%(2*np.pi)
        ###### should be real outline? requiring a loop over f or just Aout(1+ecc)######
        notdisk = (tr > self.Aout*(1.+self.ecc)) | (tr < self.Ain*(1-self.ecc))  # - individual grid elements not in disk
        xydisk =  tr[:,:,0] <= self.Aout*(1.+self.ecc)+Smax*self.sinthet  # - tracing outline of disk on observer xy plane
        self.r = tr

        
        

        #print "new grid {t}".format(t=time.clock()-tst)
        # Here include section that redefines S along the line of sight
        # (for now just use the old grid)

        # interpolate to calculate disk temperature and densities
        #print 'interpolating onto radiative transfer grid'
        #need to interpolate tempg from the 2-d rcf,zcf onto 3-d tr
        # x is xy plane, y is z axis
        ###### rf is 2d, zf is still 1d ######
        #xind = np.interp(tr.flatten(),self.rf,range(self.nrc)) #rf,nrc
        #yind = np.interp(np.abs(tdiskZ).flatten(),self.zf,range(self.nzc)) #zf,nzc
        #indices in structure arrays of coordinates in transform grid`
        zind = np.interp(np.abs(tdiskZ).flatten(),self.zf,range(self.nzc)) #zf,nzc
        phiind = np.interp(tphi.flatten(),self.pf,range(self.nphi))
        aind = np.interp((tr.flatten()*(1+self.ecc*np.cos(tphi.flatten()-self.aop)))/(1.-self.ecc**2),self.af,range(self.nac),right=self.nac)
        
        
        #print "index interp {t}".format(t=time.clock()-tst)
        ###### fixed T,Omg,rhoG still need to work on zpht ######
        tT = ndimage.map_coordinates(self.tempg,[[aind],[phiind],[zind]],order=1,cval=1e-18).reshape(self.nphi,self.nr,self.nz) #interpolate onto coordinates xind,yind #tempg
        #Omgx = ndimage.map_coordinates(self.Omg0[0],[[aind],[phiind],[zind]],order=1,cval=1e-18).reshape(self.nphi,self.nr,self.nz) #Omgs
        #Omg = ndimage.map_coordinates(self.Omg0,[[aind],[phiind],[zind]],order=1,cval=1e-18).reshape(self.nphi,self.nr,self.nz) #Omgy
        tvel = ndimage.map_coordinates(self.vel,[[aind],[phiind],[zind]],order=1).reshape(self.nphi,self.nr,self.nz)
        #Omgz = np.zeros(np.shape(Omgy))
        #trhoG = Disk.H2tog*self.Xmol/Disk.m0*ndimage.map_coordinates(self.rho0,[[aind],[phiind],[zind]],order=1,cval=1e-18).reshape(self.nphi,self.nr,self.nz)
        #trhoH2 = trhoG/self.Xmol #** not on cluster**
        #zpht = np.interp(tr.flatten(),self.rf,self.zpht).reshape(self.nphi,self.nr,self.nz) #tr,rf,zpht
        tsig_col = ndimage.map_coordinates(self.sig_col,[[aind],[phiind],[zind]],order=1,cval=1e-18).reshape(self.nphi,self.nr,self.nz)
        zpht_up = ndimage.map_coordinates(self.zpht_up,[[aind],[phiind]],order=1).reshape(self.nphi,self.nr,self.nz) #tr,rf,zpht
        zpht_low = ndimage.map_coordinates(self.zpht_low,[[aind],[phiind]],order=1).reshape(self.nphi,self.nr,self.nz) #tr,rf,zpht
        tT[notdisk] = 0 
        self.sig_col = tsig_col

        self.add_mol_ring(self.Rabund[0]/Disk.AU,self.Rabund[1]/Disk.AU,self.sigbound[0]/Disk.sc,self.sigbound[1]/Disk.sc,self.Xco,initialize=True)

        if np.size(self.Xco)>1:
            Xmol = self.Xco[0]*np.exp(-(self.Rabund[0]-tr)**2/(2*self.Rabund[3]**2))+self.Xco[1]*np.exp(-(self.Rabund[1]-tr)**2/(2*self.Rabund[4]**2))+self.Xco[2]*np.exp(-(self.Rabund[2]-tr)**2/(2*self.Rabund[5]**2))
        #else:
        #    Xmol = self.Xco



        #print "image interp {t}".format(t=time.clock()-tst)
        
        # photo-dissociation
        #zap = (np.abs(tdiskZ) > zpht_up)
        #if zap.sum() > 0:
        #    trhoG[zap] = 1e-18*trhoG[zap]
        #zap = (np.abs(tdiskZ) < zpht_low)
        #if zap.sum()>0:
        #    trhoG[zap] = 1e-18*trhoG[zap]

        #if np.size(self.Xco)<2:
        #    #Inner and outer abundance boundaries
        #    zap = (tr<=self.Rabund[0]) | (tr>=self.Rabund[1])
        #    if zap.sum()>0:
        #        trhoG[zap] = 1e-18*trhoG[zap]

        # freeze out
        zap = (tT <= Disk.Tco)
        if zap.sum() >0:
            self.Xmol[zap] = 1e-8*self.Xmol[zap]
            #trhoG[zap] = 1e-8*trhoG[zap]

        trhoH2 = Disk.H2tog/Disk.m0*ndimage.map_coordinates(self.rho0,[[aind],[phiind],[zind]],order=1,cval=1e-18).reshape(self.nphi,self.nr,self.nz)
        trhoG = trhoH2*self.Xmol 
        trhoH2[notdisk] = 0
        trhoG[notdisk] = 0
        self.rhoH2 = trhoH2

        #print "zap {t}".format(t=time.clock()-tst)
        #temperature and turbulence broadening
        #moved this to the set_line method
        #tdBV = np.sqrt(2.*Disk.kB/(Disk.Da*Disk.mCO)*tT+self.vturb**2)
        #tdBV = np.sqrt((1+(self.vturb/Disk.kms)**2.)*(2.*Disk.kB/(Disk.Da*Disk.mCO)*tT)) #vturb proportional to cs

        
        if 0:
            print 'plotting'
            plt.figure(1)
            plt.subplot(211)
            plt.pcolor(np.log10(tr[0,:,:]),tdiskZ[0,:,:],np.log10(trhoG[0,:,:]))
            plt.colorbar()
            #plt.subplot(212)
            #plt.pcolor(self.rf[:,0,np.newaxis]*np.ones(256*5),(self.zf[:,np.newaxis]*np.ones(256)).T,np.log10(self.rho0[:,0,:])) #need to expand rf and zf to same dimensions as tempg
            #plt.colorbar()
            plt.show()

        # store disk
        self.X = X
        self.Y = Y
        self.Z = tdiskZ
        self.S = S
        #self.r = tr
        self.T = tT
        #self.dBV = tdBV
        self.rhoG = trhoG
        #self.Omg = Omg#Omgy #need to combine omgx,y,z
        self.vel = tvel
        self.i_notdisk = notdisk
        self.i_xydisk = xydisk
        #self.rhoH2 = trhoH2 #*** not on cluster ***
        #self.sig_col=tsig_col
        #self.Xmol = Xmol
        self.cs = np.sqrt(2*self.kB/(self.Da*2)*self.T)
        #self.tempg = tempg
        #self.zpht = zpht
        #self.phi = tphi

    def set_line(self,line='co',vcs=True):
        if line.lower()[:2]=='co':
            m_mol = 12.011+15.999
        elif line.lower()[:4]=='c18o':
            m_mol = 12.011+17.999
        elif line.lower()[:4]=='13co':
            m_mol = 13.003+15.999
        else:
            #assume it is DCO+
            m_mol = 2.014+12.011+15.999
        if vcs:
            #temperature and turbulence broadening
            #tdBV = np.sqrt(2.*Disk.kB/(Disk.Da*Disk.mHCO)*tT+self.vturb**2)
            tdBV = np.sqrt((1+(self.vturb/Disk.kms)**2.)*(2.*Disk.kB/(Disk.Da*m_mol)*self.T)) #vturb proportional to cs

        else: #assume line.lower()=='co'
            #temperature and turbulence broadening
            tdBV = np.sqrt(2.*Disk.kB/(Disk.Da*m_mol)*tT+self.vturb**2)
            #tdBV = np.sqrt((1+(self.vturb/Disk.kms)**2.)*(2.*Disk.kB/(Disk.Da*Disk.mCO)*self.T)) #vturb proportional to cs

        self.dBV=tdBV


    def add_dust_ring(self,Rin,Rout,dtg,ppD,initialize=False):
        '''Add a ring of dust with a specified inner radius, outer radius, dust-to-gas ratio (defined at the midpoint) and slope of the dust-to-gas-ratio'''
        
        if initialize:
            self.dtg = 0*self.r
            self.kap = 2.3
        
        w = (self.r>(Rin*Disk.AU)) & (self.r<(Rout*Disk.AU))
        Rmid = (Rin+Rout)/2.*Disk.AU
        self.dtg[w] += dtg*(self.r[w]/Rmid)**(-ppD)
        self.rhoD = self.rhoH2*self.dtg*2*Disk.mh

    def add_mol_ring(self,Rin,Rout,Sig0,Sig1,abund,alpha=0,initialize=False,just_frozen=False):
        ''' Add a ring of fixed abundance, between Rin and Rout (in the radial direction) and Sig0 and Sig1 (in the vertical direction). The abundance is treated as a power law in the radial direction, with alpha as the power law exponent, and normalized at the inner edge of the ring (abund~abund0*(r/Rin)^(alpha))
        disk.add_mol_ring(10,100,.79,1000,1e-4)
        just_frozen: only apply the abundance adjustment to the areas of the disk where CO is nominally frozen out.'''
        if initialize:
            self.Xmol = np.zeros(np.shape(self.r))+1e-18
        if just_frozen:
            add_mol = (self.sig_col*Disk.Hnuctog/Disk.m0>Sig0*Disk.sc) & (self.sig_col*Disk.Hnuctog/Disk.m0<Sig1*Disk.sc) & (self.r>Rin*Disk.AU) & (self.r<Rout*Disk.AU) & (self.T<self.Tco)
        else:
            add_mol = (self.sig_col*Disk.Hnuctog/Disk.m0>Sig0*Disk.sc) & (self.sig_col*Disk.Hnuctog/Disk.m0<Sig1*Disk.sc) & (self.r>Rin*Disk.AU) & (self.r<Rout*Disk.AU)
        if add_mol.sum()>0:
            self.Xmol[add_mol]+=abund*(self.r[add_mol]/(Rin*Disk.AU))**(alpha)
        #add soft boundaries
        edge1 = (self.sig_col*Disk.Hnuctog/Disk.m0>Sig0*Disk.sc) & (self.sig_col*Disk.Hnuctog/Disk.m0<Sig1*Disk.sc) & (self.r>Rout*Disk.AU)
        if edge1.sum()>0:
            self.Xmol[edge1] += abund*(self.r[edge1]/(Rin*Disk.AU))**(alpha)*np.exp(-(self.r[edge1]/(Rout*Disk.AU))**16)
        edge2 = (self.sig_col*Disk.Hnuctog/Disk.m0>Sig0*Disk.sc) & (self.sig_col*Disk.Hnuctog/Disk.m0<Sig1*Disk.sc) & (self.r<Rin*Disk.AU)
        if edge2.sum()>0:
            self.Xmol[edge2] += abund*(self.r[edge2]/(Rin*Disk.AU))**(alpha)*(1-np.exp(-(self.r[edge2]/(Rin*Disk.AU))**20.))
        edge3 = (self.sig_col*Disk.Hnuctog/Disk.m0<Sig0*Disk.sc) & (self.r>Rin*Disk.AU) & (self.r<Rout*Disk.AU)
        if edge3.sum()>0:
            self.Xmol[edge3] += abund*(self.r[edge3]/(Rin*Disk.AU))**(alpha)*(1-np.exp(-((self.sig_col[edge3]*Disk.Hnuctog/Disk.m0)/(Sig0*Disk.sc))**8.))
        zap = (self.Xmol<0)
        if zap.sum()>0:
            self.Xmol[zap]=1e-18
        if not initialize:
            self.rhoG = self.rhoH2*self.Xmol
        
       

    def calc_hydrostatic(self,tempg,siggas,grid):
        nac = grid['nac']
        nfc = grid['nfc']
        nzc = grid['nzc']
        rcf = grid['rcf']
        zcf = grid['zcf']
        dz = (zcf - np.roll(zcf,1))#,axis=2))

        #compute rho structure
        rho0 = np.zeros((nac,nfc,nzc))
        sigint = siggas

        #compute gravo-thermal constant
        grvc = Disk.G*self.Mstar*Disk.m0/Disk.kB
        
        #t1 = time.clock()
        #differential equation for vertical density profile
        dlnT = (np.log(tempg)-np.roll(np.log(tempg),1,axis=2))/dz
        dlnp = -1.*grvc*zcf/(tempg*(rcf**2+zcf**2)**1.5)-dlnT
        dlnp[:,:,0] = -1.*grvc*zcf[:,:,0]/(tempg[:,:,0]*(rcf[:,:,0]**2.+zcf[:,:,0]**2.)**1.5)

        #numerical integration to get vertical density profile
        foo = dz*(dlnp+np.roll(dlnp,1,axis=2))/2.
        foo[:,:,0] = np.zeros((nac,nfc))
        lnp = foo.cumsum(axis=2)

        #normalize the density profile (note: this is just half the sigma value!)
        rho0 = 0.5*((sigint/np.trapz(np.exp(lnp),zcf,axis=2))[:,:,np.newaxis]*np.ones(nzc))*np.exp(lnp)
        #t2=time.clock()
        #print "hydrostatic loop took {t} seconds".format(t=(t2-t1))

        #print 'Doing hydrostatic equilibrium'
        #t1 = time.clock()
        #for ia in range(nac):
        #    for jf in range(nfc):
        #        
        #        #extract the T(z) profile at a given radius
        #        T = tempg[ia,jf]
        #                
        #        z=zcf[ia,jf]
        #        #differential equation for vertical density profile
        #        dlnT = (np.log(T)-np.roll(np.log(T),1))/dz[ia,jf]
        #        dlnp = -1*grvc*z/(T*(rcf[ia,jf]**2.+z**2.)**1.5)-dlnT
        #        dlnp[0] = -1*grvc*z[0]/(T[0]*(rcf[ia,jf,0]**2.+z[0]**2.)**1.5)
        #        
        #        #numerical integration to get vertical density profile
        #        foo = dz[ia,jf]*(dlnp+np.roll(dlnp,1))/2.
        #        foo[0] = 0.
        #        lnp = foo.cumsum()
        #
        #        #normalize the density profile (note: this is just half the sigma value!)
        #        #print lnp.shape,grvc.shape,z.shape,T.shape,rcf[ia,jf].shape,dlnT.shape
        #        dens = 0.5*sigint[ia,jf]*np.exp(lnp)/np.trapz(np.exp(lnp),z)
        #        rho0[ia,jf,:] = dens
        #        #if ir == 200:
        #        #    plt.plot(z/Disk.AU,dlnT)
        #        #    plt.plot(z/Disk.AU,dlnp)
        #t2=time.clock()
        #print "hydrostatic loop took {t} seconds".format(t=(t2-t1))

        self.rho0=rho0
        #print Disk.G,self.Mstar,Disk.m0,Disk.kB
        if 0:
            print 'plotting'
            plt.pcolor(rcf[:,0,np.newaxis]*np.ones(nzc),zcf[:,0,:],np.log10(rho0[:,0,:]))
            plt.colorbar()
            plt.show()
            
    def set_scale_height(self, sh_relation, rcf):
        'set scale height parameter using given relationship'

        if sh_relation.lower() == 'linear':
            self.H = self.sh_param * rcf
        elif sh_relation.lower() == 'const':
            length = np.ones(len(rcf)) * Disk.AU
            self.H = self.sh_param * length
        else:
            print 'WARNING::Could not determine scale height structure from given inputs. Please check sh_relation.'


    def density(self):
        'Return the density structure'
        return self.rho0

    def temperature(self):
        'Return the temperature structure'
        return self.tempg

    def grid(self):
        'Return an XYZ grid (but which one??)'
        return self.grid

    def get_params(self):
        params=[]
        params.append(self.qq)
        params.append(self.McoG/Disk.Msun)
        params.append(self.pp)
        params.append(self.Ain/Disk.AU)
        params.append(self.Aout/Disk.AU)
        params.append(self.Rc/Disk.AU)
        params.append(math.degrees(self.thet))
        params.append(self.Mstar/Disk.Msun)
        params.append(self.Xco)
        params.append(self.vturb/Disk.kms)
        params.append(self.zq0)
        params.append(self.tmid0)
        params.append(self.tatm0)
        params.append(self.handed)
        return params

    def get_obs(self):
        obs = []
        obs.append(self.nr)
        obs.append(self.nphi)
        obs.append(self.nz)
        obs.append(self.zmax/Disk.AU)
        return obs

    def plot_structure(self,sound_speed=False,beta=None,dust=False,rmax=500,zmax=170):
        ''' Plot temperature and density structure of the disk'''
        plt.figure()
        plt.rc('axes',lw=2)
        cs2 = plt.contour(self.r[0,:,:]/Disk.AU,self.Z[0,:,:]/Disk.AU,np.log10(self.rhoG[0,:,:])+4,np.arange(0,11,0.1))
        cs2 = plt.contour(-self.r[0,:,:]/Disk.AU,self.Z[0,:,:]/Disk.AU,np.log10(self.rhoG[self.nphi/2,:,:])+4,np.arange(0,11,0.1))
        ax = plt.gca()
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(14)
            tick.label1.set_fontweight('bold')
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(14)
            tick.label1.set_fontweight('bold')
        if sound_speed:
            cs = self.r*self.Omg#np.sqrt(2*self.kB/(self.Da*self.mCO)*self.T)
            cs3 = plt.contour(self.r[0,:,:]/Disk.AU,self.Z[0,:,:]/Disk.AU,cs[0,:,:]/Disk.kms,100,colors='k')
            cs3 = plt.contour(-self.r[0,:,:]/Disk.AU,self.Z[0,:,:]/Disk.AU,cs[self.nphi/2.,:,:]/Disk.kms,100,colors='k')
            plt.clabel(cs3)
        elif beta is not None:
            cs = np.sqrt(2*self.kB/(self.Da*self.mu)*self.T)
            rho = (self.rhoG+4)*self.mu*self.Da #mass density
            Bmag = np.sqrt(8*np.pi*rho*cs**2/beta) #magnetic field
            cs3 = plt.contour(self.r[0,:,:]/Disk.AU,self.Z[0,:,:]/Disk.AU,np.log10(Bmag[0,:,:]),20)
            cs3 = plt.contour(-self.r[0,:,:]/Disk.AU,self.Z[0,:,:]/Disk.AU,np.log10(Bmag[self.nphi/2.,:,:]),20)
        elif dust:
            cs3 = plt.contour(self.r[0,:,:]/Disk.AU,self.Z[0,:,:]/Disk.AU,np.log10(self.rhoD[0,:,:]),100,colors='k',linestyles='--')
            cs3 = plt.contour(-self.r[0,:,:]/Disk.AU,self.Z[0,:,:]/Disk.AU,np.log10(self.rhoD[self.nphi/2.,:,:]),100,colors='k',linestyles='--')
        else:
            cs3 = plt.contour(self.r[0,:,:]/Disk.AU,self.Z[0,:,:]/Disk.AU,self.T[0,:,:],(20,40,60,80,100,120),colors='k',ls='--')
            cs3 = plt.contour(-self.r[0,:,:]/Disk.AU,self.Z[0,:,:]/Disk.AU,self.T[self.nphi/2.,:,:],(20,40,60,80,100,120),colors='k',ls='--')
            plt.clabel(cs3,fmt='%1i')
        plt.colorbar(cs2,label='log n')
        plt.xlim(-1*rmax,rmax)
        plt.ylim(0,zmax)
        plt.xlabel('R (AU)',fontsize=20)
        plt.ylabel('Z (AU)',fontsize=20)
        plt.show()

    def calcH(self,verbose=True):
        ''' Calculate the equivalent of the pressure scale height within our disks. This is useful for comparison with other models that take this as a free parameter. H is defined as 2^(-.5) times the height where the density drops by 1/e. (The factor of 2^(-.5) is included to be consistent with a vertical density distribution that falls off as exp(-z^2/2H^2))'''
        ###### this method does not work with the elliptical disk (must expand to 3d) ######
        nrc = self.nrc
        zf = self.zf
        rf = self.rf
        rho0 = self.rho0

        H = np.zeros(nrc)
        for i in range(nrc):
            rho_cen = rho0[i,0]
            diff = abs(rho_cen/np.e-rho0[i,:])
            H[i] = zf[(diff == diff.min())]/np.sqrt(2.)

        if verbose:
            H100 = np.interp(100*Disk.AU,rf,H)
            psi = (np.polyfit(np.log10(rf),np.log10(H),1))[0]
            #print H100/Disk.AU
            #print psi
            print 'H100 (AU): {:.3f}'.format(H100/Disk.AU)
            print 'power law: {:.3f}'.format(psi)

        return H

    