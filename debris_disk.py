# Define the Disk class. This class of objects can be used to create a disk structure. Given 
#parameters defining the disk, it calculates the dust density structure using a simple radial 
#power-law relation and defines the grid used for radiative transfer. This object can then be 
#fed into the modelling code which does the radiative transfer given this structure.

#two methods for creating an instance of this class

# from disk import *
# x=Disk()

# import disk
# x = disk.Disk()

# For testing purposes use the second method. Then I can use reload(disk) after updating the code.
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy import ndimage
from astropy import constants as const

class Disk:
    
    'Common class for circumstellar disk structure'
    
    #Define useful constants
    AU      = const.au.cgs.value       # - astronomical unit (cm)
    Rsun    = const.R_sun.cgs.value    # - radius of the sun (cm)
    c       = const.c.cgs.value        # - speed of light (cm/s)
    h       = const.h.cgs.value        # - Planck's constant (erg/s)
    kB      = const.k_B.cgs.value      # - Boltzmann's constant (erg/K)
    sigmaB  = const.sigma_sb.cgs.value # - Stefan-Boltzmann constant (erg cm^-2 s^-1 K^-4)
    pc      = const.pc.cgs.value       # - parsec (cm)
    Jy      = 1.e23                    # - cgs flux density (Janskys)
    Lsun    = const.L_sun.cgs.value    # - luminosity of the sun (ergs)
    Mearth  = const.M_earth.cgs.value  # - mass of the earth (g)
    mh      = const.m_p.cgs.value      # - proton mass (g)
    Da      = mh                       # - atmoic mass unit (g)
    Msun    = const.M_sun.cgs.value    # - solar mass (g)
    G       = const.G.cgs.value        # - gravitational constant (cm^3/g/s^2)
    rad     = 206264.806               # - radian to arcsecond conversion
    kms     = 1e5                      # - convert km/s to cm/s
    GHz     = 1e9                      # - convert from GHz to Hz
    mCO     = 12.011+15.999            # - CO molecular weight
    mDCO    = mCO+2.014                # - HCO molecular weight
    mu      = 2.37                     # - gas mean molecular weight
    m0      = mu*mh                    # - gas mean molecular opacity
    Hnuctog = 0.706*mu                 # - H nuclei abundance fraction (H nuclei:gas)
    sc      = 1.59e21                  # - Av --> H column density (C. Qi 08,11)
    H2tog   = 0.8                      # - H2 abundance fraction (H2:gas)
    Tco     = 19.                      # - freeze out
    sigphot = 0.79*sc                  # - photo-dissociation column
    
    def __init__(self,params=[-0.5,0.09,1.,10.,1000.,150.,51.5,2.3,1e-4,0.01,33.9,19.,69.3,
    	         [.79,1000],[10.,1000],-1, 500, 500, 0.09, 0.1],obs=[180,131,300,170],rtg=True,
                 vcs=True,line='co',ring=None):

        self.ring=ring
        self.set_obs(obs)   # set the observational parameters
        self.set_params(params) # set the structure parameters

        self.set_structure()  # use obs and params to create disk structure
        if rtg:
            self.set_rt_grid(vcs=vcs)
            self.set_line(line=line,vcs=vcs)

    def set_params(self,params):
        'Set the disk structure parameters'
        self.qq = params[0]                 # - temperature index
        self.Mdust = params[1]*Disk.Msun    # - dust mass
        self.pp = params[2]                 # - surface density index
        self.Rin = params[3]*Disk.AU        # - inner edge in cm
        self.Rout = params[4]*Disk.AU       # - outer edge in cm
        self.Rc = params[5]*Disk.AU         # - critical radius in cm
        self.thet = math.radians(params[6]) # - convert inclination to radians
        self.Mstar = params[7]*Disk.Msun    # - convert mass of star to g
        self.Xco = params[8]                # - CO gas fraction
        self.vturb = params[9]*Disk.kms     # - turbulence velocity
        self.zq0 = params[10]               # - Zq, in AU, at 150 AU
        self.sigbound = [params[11][0]*Disk.sc,params[11][1]*Disk.sc]          
              # - upper and lower column density boundaries
        if len(params[12]) ==2:
            # - inner and outer abundance boundaries 
            self.Rabund = [params[12][0]*Disk.AU,params[12][1]*Disk.AU]
        else:
            # - inner/outer ring, width of inner/outer ring
            self.Rabund=[params[12][0]*Disk.AU,params[12][1]*Disk.AU,params[12][2]*Disk.AU,params[12][3]*Disk.AU,params[12][4]*Disk.AU,params[12][5]*Disk.AU]
        self.handed = params[13]            # 
        self.costhet = np.cos(self.thet)  # - cos(i)
        self.sinthet = np.sin(self.thet)  # - sin(i)
        if self.ring is not None:
            self.Rring = self.ring[0]*Disk.AU # location of the ring
            self.Wring = self.ring[1]*Disk.AU #width of ring
            self.sig_enhance = self.ring[2] #power law slope of inner temperature structure

        #set number of model grid elements in radial and vertical direction
        self.r_gridsize = params[14]
        self.z_gridsize = params[15]
        self.Lstar      = params[16]
        self.sh_param   = params[17]
        
    def set_obs(self,obs):
        'Set the observational parameters. These parameters are the number of r, phi, S grid points in the radiative transer grid, along with the maximum height of the grid.'
        self.nr = obs[0]
        self.nphi = obs[1]
        self.nz = obs[2]
        self.zmax = obs[3]*Disk.AU
                

    def set_structure(self):
        '''Calculate the disk density and temperature structure given the specified parameters'''
        # Define the desired regular cylindrical (r,z) grid
        nrc  = self.r_gridsize  # - number of unique r points
        nzc  = self.z_gridsize  # - number of unique z points
        rmin = self.Rin         # - minimum r [AU]

        rmax = self.Rout        # - maximum r [AU]
        zmin = .1*Disk.AU       # - minimum z [AU] *****0.1?
        rf   = np.logspace(np.log10(rmin),np.log10(rmax),nrc)
        zf   = np.logspace(np.log10(zmin),np.log10(self.zmax),nzc)

        idr  = np.ones(nrc)
        zcf  = np.outer(idr,zf)
        rcf  = rf[:,np.newaxis]*np.ones(nzc)

        # Interpolate dust temperature and density onto cylindrical grid
        tf = 0.5*np.pi-np.arctan(zcf/rcf)  # theta values
        rrf = np.sqrt(rcf**2.+zcf**2)

        # bundle the grid for helper functions
        grid = {'nrc':nrc,'nzc':nzc,'rf':rf,'rmax':rmax,'zcf':zcf}
        self.grid=grid

        #define temperature structure
        tempg = (self.Lstar * self.Lsun / (16. * np.pi * rcf**2 * self.sigmaB))**0.25
        
        #calculate the scale height of the disk
        self.H = self.sh_param * rcf

        #calculate the dust critical surface density
        dsigma_crit = self.Mdust * (self.pp + 2.) / (2. * np.pi * (self.Rout**(2. + self.pp) - self.Rin**(2. + self.pp)))

        #calculate the dust surface density structure
        self.sigmaD = dsigma_crit * (rcf**self.pp)

        #calculate the dust volume density structure as a function of radius
        rhoD = self.sigmaD / (self.H * np.sqrt(np.pi)) * np.exp(-1. * (zcf / self.H)**2)

        #plt.plot(zcf[497,:]/Disk.AU, rhoD[497,:])
        
        #plt.xlabel(r"$z$($r$ = 40 [au]) [au]")
        #plt.ylabel(r"$\rho_D (z)$ [g cm$^{-3}$]")
        #plt.title("Mass Density vs. Midplane Distance at 40 au")
        #plt.show()

        # Calculate vertical density structure
        #Sc_old = self.Mdust*(2.-self.pp)/(2*np.pi*self.Rc*self.Rc)
                         #        siggas = Sc*(rf/self.Rc)**(-1*self.pp)*np.exp(-1*(rf/self.Rc)**(2-self.pp))
        #change the previous line to be a power law in radius, with index pp, extending from Rin to Rc. From Rc to Rout the surface density is 0 (will the zeros cause problems for the hydrostatic equilibrium calculation? if it does then I could simply set Rout=Rc)
        #what is normalization on Sig_gas, in terms of Mdisk?
        #if self.pp ==2:
        #    Sc = self.Mdust/(2*np.pi*(np.log(self.Rc)-np.log(self.Rin)))
        #else:
        #    Sc = self.Mdust*(2.-self.pp)/(2*np.pi*(self.Rc**(2-self.pp)-self.Rin**(2-self.pp)))

        #siggas = Sc*rf**(-1*self.pp)
        #siggas[rf<self.Rin] = 1e-60
        #siggas[rf>self.Rc] = 1e-60

        #if self.ring is not None:
        #    w = np.abs(rcf-self.Rring)<self.Wring/2.
        #    if w.sum()>0:
        #        tempg[w] = tempg[w]*(rcf[w]/(150*Disk.AU))**(self.sig_enhance-self.qq)((rcf[w].max())/(150.*Disk.AU))**(-self.qq+self.enhance)


        #self.calc_hydrostatic(tempg,siggas,grid)
        
        #Calculate radial pressure differential
        #Pgas = Disk.kB/Disk.m0*self.rho0*tempg
        #dPdr = (np.roll(Pgas,-1,axis=0)-Pgas)/(np.roll(rcf,-1,axis=0)-rcf)
        
        #Calculate velocity field
        #Omg = np.sqrt((dPdr/(rcf*self.rho0)+Disk.G*self.Mstar/(rcf**2+zcf**2)**1.5))
        #Omk = np.sqrt(Disk.G*self.Mstar/rcf**3.)
        
        # Check for NANs
        #ii = np.isnan(Omg)
        #Omg[ii] = Omk[ii]
        ii = np.isnan(rhoD)
        if ii.sum() > 0:
            rhoD[ii] = 1e-60
            print 'Beware: removed NaNs from dust density (#%s)' % ii.sum()
        ii = np.isnan(tempg)
        if ii.sum() > 0:
            tempg[ii] = 2.73
            print 'Beware: removed NaNs from temperature (#%s)' % ii.sum()

        # find photodissociation boundary layer from top
        sig_col = np.zeros((nrc,nzc)) #Cumulative mass surface density along vertical lines starting at z=170AU
        self.sig_col = sig_col #save it for later
        
        # Set default freeze-out temp
        self.Tco = Disk.Tco


        
        self.rf     = rf
        self.nrc    = nrc
        self.zf     = zf
        self.nzc    = nzc
        self.tempg  = tempg
        self.rhoD   = rhoD
        self.rhoD0  = rhoD
        #self.Omg0   = Omg
        
        
        
        
    def set_rt_grid(self,vcs=True):
        ### Start of Radiative Transfer portion of the code...
        # Define and initialize cylindrical grid
        Smin = 1*Disk.AU                 # offset from zero to log scale
        if self.thet > np.arctan(self.Rout/self.zmax):
            Smax = 2*self.Rout/self.sinthet
        else:
            Smax = 2.*self.zmax/self.costhet       # los distance through disk
        Smid = Smax/2.                    # halfway along los
        ytop = Smax*self.sinthet/2.       # y origin offset for observer xy center

        R = np.linspace(0,self.Rout,self.nr) 
        phi = np.arange(self.nphi)*2*np.pi/(self.nphi-1)
        foo = np.floor(self.nz/2)
        
        S_old = np.arange(2*foo)/(2*foo)*(Smax-Smin)+Smin
        
        
        # Basically copy S_old, with length nz,  into each column of a nphi*nr*nz matrix
        S = (S_old[:,np.newaxis,np.newaxis]*np.ones((self.nr,self.nphi))).T

        # arrays in [phi,r,s]
        X = (np.outer(R,np.cos(phi))).transpose()
        Y = (np.outer(R,np.sin(phi))).transpose()

        zsky = Smid-S
        tdiskZ = (Y.repeat(self.nz).reshape(self.nphi,self.nr,self.nz))*self.sinthet+zsky*self.costhet
        tdiskY = (Y.repeat(self.nz).reshape(self.nphi,self.nr,self.nz))*self.costhet-zsky*self.sinthet

        # transform grid
#        tdiskZ = self.zmax*(np.ones((self.nphi,self.nr,self.nz)))-self.costhet*S
#        if self.thet > np.arctan(self.Rout/self.zmax):
#            tdiskZ -=(Y*self.sinthet).repeat(self.nz).reshape(self.nphi,self.nr,self.nz)
#        tdiskY = ytop - self.sinthet*S + (Y/self.costhet).repeat(self.nz).reshape(self.nphi,self.nr,self.nz)
        tr = np.sqrt(X.repeat(self.nz).reshape(self.nphi,self.nr,self.nz)**2+tdiskY**2)
        notdisk = (tr > self.Rout) | (tr < self.Rin)  # - individual grid elements not in disk
        xydisk =  tr[:,:,0] <= self.Rout+Smax*self.sinthet  # - tracing outline of disk on observer xy plane
        

        # interpolate to calculate disk temperature and densities
        #print 'interpolating onto radiative transfer grid'
        #need to interpolate tempg from the 2-d rcf,zcf onto 3-d tr
        xind = np.interp(tr.flatten(),self.rf,range(self.nrc)) #rf,nrc
        yind = np.interp(np.abs(tdiskZ).flatten(),self.zf,range(self.nzc)) #zf,nzc
        tT = ndimage.map_coordinates(self.tempg,[[xind],[yind]],order=1).reshape(self.nphi,self.nr,self.nz) #interpolate onto coordinates xind,yind #tempg
        tDD = ndimage.map_coordinates(self.rhoD,[[xind],[yind]],order=1).reshape(self.nphi,self.nr,self.nz) #interpolate onto coordinates xind,yind #dustg
        #Omg = ndimage.map_coordinates(self.Omg0,[[xind],[yind]],order=1).reshape(self.nphi,self.nr,self.nz) #Omg
        tsig_col = ndimage.map_coordinates(self.sig_col,[[xind],[yind]],order=2).reshape(self.nphi,self.nr,self.nz)

        tT[notdisk]=2.73
        self.r = tr
        self.sig_col = tsig_col

        #Set molecular abundance
        self.add_mol_ring(self.Rabund[0]/Disk.AU,self.Rabund[1]/Disk.AU,self.sigbound[0]/Disk.sc,self.sigbound[1]/Disk.sc,self.Xco,initialize=True)

        if np.size(self.Xco)>1:
            #gaussian rings
            self.Xmol = self.Xco[0]*np.exp(-(self.Rabund[0]-tr)**2/(2*self.Rabund[3]**2))+self.Xco[1]*np.exp(-(self.Rabund[1]-tr)**2/(2*self.Rabund[4]**2))+self.Xco[2]*np.exp(-(self.Rabund[2]-tr)**2/(2*self.Rabund[5]**2))

        #Freeze-out
        zap = (tT<self.Tco)
        if zap.sum() > 0:
            self.Xmol[zap] = 1/5.*self.Xmol[zap]

        #trhoH2 = Disk.H2tog/Disk.m0*ndimage.map_coordinates(self.rho0,[[xind],[yind]],order=1).reshape(self.nphi,self.nr,self.nz)
        #trhoG = trhoH2*self.Xmol
        #trhoG[notdisk] = 0
        #trhoH2[notdisk] = 0
        #self.rhoH2 = trhoH2

        self.add_dust_ring(self.Rin,self.Rout,0.,0.,initialize=True) #initialize dust density to 0

        ##from scipy.special import erfc
        ##trhoG = .5*erfc((tdiskZ-zpht_up)/(.1*zpht_up))*trhoG

        #temperature and turbulence broadening
        #tdBV = np.sqrt(2.*Disk.kB/(Disk.Da*Disk.mCO)*tT+self.vturb**2)
        if vcs:
            tdBV = np.sqrt((1+(self.vturb/Disk.kms)**2.)*(2.*Disk.kB/(Disk.Da*Disk.mCO)*tT)) #vturb proportional to cs
            #self.r = tr
            #self.Z = tdiskZ
            #vt = self.doppler()*np.sqrt(Disk.Da*Disk.mCO/(Disk.Da*Disk.mu))
            #tdBV = np.sqrt((1+vt**2.)*(2.*Disk.kB/(Disk.Da*Disk.mCO)*tT))
        else:
            tdBV = np.sqrt(2.*Disk.kB/(Disk.Da*Disk.mCO)*tT+self.vturb**2)

        
        # store disk
        self.X = X
        self.Y = Y
        self.Z = tdiskZ
        self.S = S
        #self.r = tr
        self.T = tT
        self.dBV = tdBV
        #self.rhoG = trhoG
        #self.Omg = Omg
        self.i_notdisk = notdisk
        self.i_xydisk = xydisk
        self.cs = np.sqrt(2*self.kB/(self.Da*2)*self.T)
        self.rhoD = tDD
        #self.sig_col=tsig_col
        #self.rhoH2 = trhoH2

    def set_line(self,line='co',vcs=True):
        if line.lower()[:2]=='co':
            m_mol = 12.011+15.999
        elif line.lower()[:4]=='c18o':
            m_mol = 12.011+17.999
        elif line.lower()[:4]=='13co':
            m_mol = 13.003+15.999
        else:
            #assume it is DCO+
            m_mol = Disk.mDCO
        if vcs:
            #temperature and turbulence broadening
            #tdBV = np.sqrt(2.*Disk.kB/(Disk.Da*Disk.mHCO)*tT+self.vturb**2)
            tdBV = np.sqrt((1+(self.vturb/Disk.kms)**2.)*(2.*Disk.kB/(Disk.Da*m_mol)*self.T)) #vturb proportional to cs

        else: #assume line.lower()=='co'
            #temperature and turbulence broadening
            tdBV = np.sqrt(2.*Disk.kB/(Disk.Da*m_mol)*tT+self.vturb**2)
            

        self.dBV=tdBV
    
    def add_dust_ring(self,Rin,Rout,dtg,ppD,initialize=True):
        '''Add a ring of dust with a specified inner radius, outer radius, dust-to-gas ratio (defined at the midpoint) and slope of the dust-to-gas-ratio'''
        
        if initialize:
            self.dtg = 0*self.r
            self.kap = 2.3
        
        w = (self.r>(Rin*Disk.AU)) & (self.r<(Rout*Disk.AU))
        Rmid = (Rin+Rout)/2.*Disk.AU
        self.dtg[w] += dtg*(self.r[w]/Rmid)**(-ppD)
        #self.rhoD = self.rhoH2*self.dtg*2*Disk.mh

    def add_mol_ring(self,Rin,Rout,Sig0,Sig1,abund,initialize=False):
        '''Add a ring of fixed abundance, between Rin and Rout (in the radial direction) and Sig0 and Sig1 (in the vertical direction).
        disk.add_mol_ring(10*disk.AU,100*disk.AU,.79*disk.sc,1000*disk.sc,1e-4)
        '''
        if initialize:
            self.Xmol = np.zeros(np.shape(self.r))+1e-18
        add_mol = (self.sig_col*Disk.Hnuctog/Disk.m0>Sig0*Disk.sc) & (self.sig_col*Disk.Hnuctog/Disk.m0<Sig1*Disk.sc) & (self.r>Rin*Disk.AU) & (self.r<Rout*Disk.AU)
        if add_mol.sum()>0:
            self.Xmol[add_mol]+=abund
        #add soft boundaries
        edge1 = (self.sig_col*Disk.Hnuctog/Disk.m0>Sig0*Disk.sc) & (self.sig_col*Disk.Hnuctog/Disk.m0<Sig1*Disk.sc) & (self.r>Rout*Disk.AU)
        if edge1.sum()>0:
            self.Xmol[edge1] += abund*np.exp(-(self.r[edge1]/(Rout*Disk.AU))**16)
        edge2 = (self.sig_col*Disk.Hnuctog/Disk.m0>Sig0*Disk.sc) & (self.sig_col*Disk.Hnuctog/Disk.m0<Sig1*Disk.sc) & (self.r<Rin*Disk.AU)
        if edge2.sum()>0:
            self.Xmol[edge2] += abund*(1-np.exp(-(self.r[edge2]/(Rin*Disk.AU))**20.))
        edge3 = (self.sig_col*Disk.Hnuctog/Disk.m0<Sig0*Disk.sc) & (self.r>Rin*Disk.AU) & (self.r<Rout*Disk.AU)
        if edge3.sum()>0:
            self.Xmol[edge3] += abund*(1-np.exp(-((self.sig_col[edge3]*Disk.Hnuctog/Disk.m0)/(Sig0*Disk.sc))**8.))
        zap = (self.Xmol<0)
        if zap.sum()>0:
            self.Xmol[zap]=1e-18
        if not initialize:
            self.rhoG = self.rhoH2*self.Xmol

    def density(self):
        'Return the density structure'
        return self.rho0

    def temperature(self):
        'Return the temperature structure'
        return self.tempg

    def dust_density(self):
        'Return the dust density structure'
        return self.rhoD0

    def grid(self):
        'Return an XYZ grid (but which one??)'
        return self.grid

    def get_params(self):
        params=[]
        params.append(self.qq)
        params.append(self.Mdust/Disk.Msun)
        params.append(self.pp)
        params.append(self.Rin/Disk.AU)
        params.append(self.Rout/Disk.AU)
        params.append(self.Rc/Disk.AU)
        params.append(math.degrees(self.thet))
        params.append(self.Mstar/Disk.Msun)
        params.append(self.Xco)
        params.append(self.vturb/Disk.kms)
        params.append(self.zq0)
        params.append(self.handed)
        params.append(self.r_gridsize)
        params.append(self.z_gridsize)
        params.append(self.Lstar)
        params.append(self.sh_param)
        return params

    def get_obs(self):
        obs = []
        obs.append(self.nr)
        obs.append(self.nphi)
        obs.append(self.nz)
        obs.append(self.zmax/Disk.AU)
        return obs

    def calcH(self,verbose=True,return_pow=False):
        ''' Calculate the equivalent of the pressure scale height within our disks. This is useful for comparison with other models that take this as a free parameter. H is defined as 2^(-.5) times the height where the density drops by 1/e. (The factor of 2^(-.5) is included to be consistent with a vertical density distribution that falls off as exp(-z^2/2H^2))'''

        nrc = self.nrc
        zf = self.zf
        rf = self.rf
        rho0 = self.rho0

        H = np.zeros(nrc)
        for i in range(nrc):
            rho_cen = rho0[i,0]
            diff = abs(rho_cen/np.e-rho0[i,:])
            H[i] = zf[(diff == diff.min())]/np.sqrt(2)

        H100 = np.interp(100*Disk.AU,rf,H)
        psi = (np.polyfit(np.log10(rf),np.log10(H),1))[0]
        
        if verbose:
            print 'H100 (AU): {:.3f}'.format(H100/Disk.AU)
            print 'power law: {:.3f}'.format(psi)

        if return_pow:
            return (H100/Disk.AU,psi)
        else:
            return H
