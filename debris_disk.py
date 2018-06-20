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
from numba import jitclass          # import the decorator
from numba import int32, float32    # import the types

spec = [
    ('value', int32),               # a simple scalar field
    ('array', float32[:]),          # an array field
]
    
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
    
    
    def __init__(self,
        params=[-0.5,0.09,1.,10.,1000.,150.,51.5,2.3,1e-4,0.01,33.9,19.,69.3, [.79,1000],[10.,1000], -1, 500, 500, 0.09, 0.1],
        obs=[180,131,300,170],
        rtg=True,
        vcs=True,sh_relation='linear',line='co',ring=None, annulus=None):

        self.ring=ring
        self.annulus = annulus
        self.set_obs(obs)   # set the observational parameters
        self.set_params(params) # set the structure parameters

        self.set_structure(sh_relation)  # use obs and params to create disk structure
        if rtg:
            self.set_rt_grid(vcs=vcs)
            self.set_line(line=line,vcs=vcs)

    def set_params(self,params):
        'Set the disk structure parameters'
        self.qq       = params[0]               # - temperature index
        self.Mdust    = params[1]*Disk.Msun     # - dust mass
        self.pp       = params[2]               # - surface density index
        self.Rin      = params[3]*Disk.AU       # - inner edge in cm
        self.Rout     = params[4]*Disk.AU       # - outer edge in cm
        self.Rc       = params[5]*Disk.AU       # - critical radius in cm
        self.thet     = math.radians(params[6]) # - convert inclination to radians
        self.Mstar    = params[7]*Disk.Msun     # - convert mass of star to g
        self.Xco      = params[8]               # - CO gas fraction
        self.vturb    = params[9]*Disk.kms      # - turbulence velocity
        self.zq0      = params[10]              # - Zq, in AU, at 150 AU
        self.sigbound = [params[11][0]*Disk.sc,params[11][1]*Disk.sc]          
        
        # - upper and lower column density boundaries
        if len(params[12]) == 2:
            # - inner and outer abundance boundaries 
            self.Rabund = [params[12][0]*Disk.AU,params[12][1]*Disk.AU]
        else:
            # - inner/outer ring, width of inner/outer ring
            self.Rabund=[params[12][0]*Disk.AU,params[12][1]*Disk.AU,params[12][2]*Disk.AU,params[12][3]*Disk.AU,params[12][4]*Disk.AU,params[12][5]*Disk.AU]
        
        self.handed  = params[13]         # - NEED COMMENT HERE 
        self.costhet = np.cos(self.thet)  # - cos(i)
        self.sinthet = np.sin(self.thet)  # - sin(i)
            
        #set number of model grid elements in radial and vertical direction
        self.r_gridsize = params[14]
        self.z_gridsize = params[15]
        self.Lstar      = params[16]
        self.sh_param   = params[17]
        
        if self.ring is not None:
            self.Rring       = self.ring[0]*Disk.AU # location of the ring
            self.Wring       = self.ring[1]*Disk.AU # width of ring
            self.sig_enhance = self.ring[2]         # power law slope of inner temperature structure

        if self.annulus is not None:
            self.annulus_Rin = self.annulus[0]*Disk.AU # location of the dust annulus
            self.annulus_Rout = self.annulus[1]*Disk.AU #width of annulus
            self.annulus_mass = self.annulus[2]*Disk.Mearth #total added mass
        
    def set_obs(self,obs):
        'Set the observational parameters. These parameters are the number of r, phi, S grid points in the radiative transer grid, along with the maximum height of the grid.'
        self.nr   = obs[0]
        self.nphi = obs[1]
        self.nz   = obs[2]
        self.zmax = obs[3]*Disk.AU
                

    def set_structure(self, sh_relation):
        '''Calculate the disk density and temperature structure given the specified parameters'''
        # Define the desired regular cylindrical (r,z) grid
        nrc  = self.r_gridsize  # - number of unique r points
        nzc  = self.z_gridsize  # - number of unique z points
        
        rmin = self.Rin if self.annulus is None \
            else np.min([self.Rin, self.annulus_Rin]) # - minimum r [AU]
        rmax = self.Rout if self.annulus is None  \
            else np.max([self.Rout, self.annulus_Rout]) # - maximum r [AU]      
        zmin = .1*Disk.AU       # - minimum z [AU] *****0.1?
        
        # import numpy as np
        # import pandas as pd
        # x = pd.Series(np.logspace(np.log10(10),np.log10(42),1000))#.loc[lambda x: (x > 14.9) & (x < 15)]
        # r.diff()
        # r = pd.Series(np.linspace(10,42,500)).loc[lambda x: (x > 15) & (x < 15.5)]
        # x.diff()
        
        rf   = np.logspace(np.log10(rmin),np.log10(rmax),nrc)
        zf   = np.logspace(np.log10(zmin),np.log10(self.zmax),nzc)

        idr  = np.ones(nrc)
        zcf  = np.outer(idr,zf)
        rcf  = rf[:,np.newaxis]*np.ones(nzc)

        # Interpolate dust temperature and density onto cylindrical grid
        tf  = 0.5*np.pi-np.arctan(zcf/rcf)  # theta values
        rrf = np.sqrt(rcf**2.+zcf**2)

        # bundle the grid for helper functions
        # grid = {'nrc':nrc,'nzc':nzc,'rf':rf,'rmax':rmax,'zcf':zcf}
        # self.grid=grid

        #define temperature structure
        tempg = (self.Lstar * self.Lsun / (16. * np.pi * rcf**2 * self.sigmaB))**0.25
        
        #calculate the scale height of the disk
        self.set_scale_height(sh_relation, rcf)

        #calculate the dust critical surface density
        dsigma_crit = self.Mdust * (self.pp + 2.) / (2. * np.pi * (self.Rout**(2. + self.pp) - self.Rin**(2. + self.pp)))

        #calculate the dust surface density structure
        self.sigmaD = np.full(rcf.shape, 1e-60)
        w = ((rcf > self.Rin) & (rcf < self.Rout))
        self.sigmaD[w] = dsigma_crit * (rcf[w]**self.pp)
        
        #adjust surface density to account for annulus
        if self.annulus is not None:
            w = ((rcf > self.annulus_Rin) & (rcf < self.annulus_Rout))
            annulus_sigma = self.annulus_mass / \
                (np.pi * (self.annulus_Rout**2 - self.annulus_Rin**2))
            self.sigmaD[w] += annulus_sigma
            
        #calculate the dust volume density structure as a function of radius
        rhoD = self.sigmaD / (self.H * np.sqrt(np.pi)) * np.exp(-1. * (zcf / self.H)**2)
        
        # Check for NANs
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
        self.sig_col = sig_col        #save it for later        
        
        self.rf          = rf
        self.nrc         = nrc
        self.zf          = zf
        self.nzc         = nzc
        self.tempg       = tempg
        self.dsigma_crit = dsigma_crit
        self.rhoD        = rhoD
        self.rhoD0       = rhoD
        
        
    def set_rt_grid(self,vcs=True):
        ### Start of Radiative Transfer portion of the code...
        # Define and initialize cylindrical grid
        rmin = self.Rin if self.annulus is None \
            else np.min([self.Rin, self.annulus_Rin]) # - minimum r [AU]
        rmax = self.Rout if self.annulus is None  \
            else np.max([self.Rout, self.annulus_Rout]) # - maximum r [AU]      
        
        Smin = 1*Disk.AU                 # offset from zero to log scale
        if self.thet > np.arctan(rmax/self.zmax):
            Smax = 2*rmax/self.sinthet
        else:
            Smax = 2.*self.zmax/self.costhet       # los distance through disk
        Smid = Smax/2.                    # halfway along los
        ytop = Smax*self.sinthet/2.       # y origin offset for observer xy center

        R   = np.linspace(0,rmax,self.nr) 
        phi = np.arange(self.nphi)*2*np.pi/(self.nphi-1)
        foo = np.floor(self.nz/2)
        
        S_old = np.arange(2*foo)/(2*foo)*(Smax-Smin)+Smin
        
        
        # Basically copy S_old, with length nz,  into each column of a nphi*nr*nz matrix
        S = (S_old[:,np.newaxis,np.newaxis]*np.ones((self.nr,self.nphi))).T

        # arrays in [phi,r,s]
        X = (np.outer(R,np.cos(phi))).transpose()
        Y = (np.outer(R,np.sin(phi))).transpose()

        #re-define disk midplane coordinates to be in line with radiative transfer grid
        zsky   = Smid-S
        tdiskZ = (Y.repeat(self.nz).reshape(self.nphi,self.nr,self.nz))*self.sinthet+zsky*self.costhet
        tdiskY = (Y.repeat(self.nz).reshape(self.nphi,self.nr,self.nz))*self.costhet-zsky*self.sinthet

        # transform grid
        tr      = np.sqrt(X.repeat(self.nz).reshape(self.nphi,self.nr,self.nz)**2+tdiskY**2)
        notdisk = (tr > rmax) | (tr < rmin)         # - individual grid elements not in disk
        if self.annulus is not None:
            #exclude elements between outer disk edge and inner annulus edge
            # if annulus extends beyond disk
            if self.Rout < self.annulus_Rin:
                notdisk = notdisk | ((tr > self.Rout) & (tr < self.annulus_Rin))
                
            #exclude elements between outer annulus edge and inner disk edge
            # if annulus extends interior to disk
            if self.Rin > self.annulus_Rout:
                notdisk = notdisk | ((tr > self.annulus_Rout) & (tr < self.Rin))
                
        xydisk  =  tr[:,:,0] <= rmax+Smax*self.sinthet  # - tracing outline of disk on observer xy plane
        

        # interpolate to calculate disk temperature and densities
        #print 'interpolating onto radiative transfer grid'
        #need to interpolate tempg from the 2-d rcf,zcf onto 3-d tr
        xind     = np.interp(tr.flatten(),self.rf,list(range(self.nrc)))             #rf,nrc
        yind     = np.interp(np.abs(tdiskZ).flatten(),self.zf,list(range(self.nzc))) #zf,nzc
        tT       = ndimage.map_coordinates(self.tempg,[[xind],[yind]],order=1).reshape(self.nphi,self.nr,self.nz) #interpolate onto coordinates xind,yind #tempg
        tsigmaD  = ndimage.map_coordinates(self.sigmaD,[[xind],[yind]],order=1).reshape(self.nphi,self.nr,self.nz)  #interpolate onto coordinates xind,yind #dustg
        tDD      = ndimage.map_coordinates(self.rhoD,[[xind],[yind]],order=1).reshape(self.nphi,self.nr,self.nz)  #interpolate onto coordinates xind,yind #dustg
        tH       = ndimage.map_coordinates(self.H,[[xind],[yind]],order=1).reshape(self.nphi,self.nr,self.nz)  #interpolate onto coordinates xind,yind #dustg
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

        self.add_dust_ring(self.Rin,self.Rout,0.,0.,initialize=True) #initialize dust density to 0

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
        self.T = tT
        self.dBV = tdBV
        self.i_notdisk = notdisk
        self.i_xydisk = xydisk
        self.cs = np.sqrt(2*self.kB/(self.Da*2)*self.T)
        self.sigmaD = tsigmaD
        self.rhoD = tDD
        self.H   = tH

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
    
    def add_dust_gap(self,Rin,Rout):
        '''Add a gap in the dust with a specified inner and outer radius to the disk,
           and re-normalize the remaining dust density structure to account for the
           missing mass'''

        #calculate the mass left after subtracting a ring of size Rin,Rout
        gap_mass       = 2 * np.pi * self.dsigma_crit * (((Rout*Disk.AU)**(2 + self.pp)) - ((Rin*Disk.AU)**(2 + self.pp))) / (self.pp + 2.)
        remaining_mass = self.Mdust - gap_mass

        #re-normalize disk density to account for missing gap mas
        norm_factor = self.Mdust / remaining_mass
        self.rhoD = self.rhoD * norm_factor
        
        #Finally, zero out density where the gap is located
        w = (self.r>(Rin*Disk.AU)) & (self.r<(Rout*Disk.AU))
        self.rhoD[w] = 0.

    def add_dust_mass(self, ring_mass, Rin, Rout):

        '''Add dust mass in a ring with a specific inner and outer radius to
           the disk.'''

        #calculate how much mass would initially be located within the radii specified by the user,
        #and add the ring mass to this initial mass
        ring_mass *= self.Mearth
        initial_mass = 2 * np.pi * self.dsigma_crit * (((Rout*Disk.AU)**(2 + self.pp)) - ((Rin*Disk.AU)**(2 + self.pp))) / (self.pp + 2.)
        added_mass   = initial_mass + ring_mass

        #calculate the normalization factor, added_mass / mass of the dust
        norm_factor = added_mass / initial_mass

        #multiply the dust within the specified radius 
        w = (self.r>(Rin*Disk.AU)) & (self.r<(Rout*Disk.AU))

        #re-calculate the dust volume density structure as a function ring radius
        self.rhoD[w] *= norm_factor


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
            
    def set_scale_height(self, sh_relation, rcf):
        'set scale height parameter using given relationship'

        if sh_relation.lower() == 'linear':
            self.H = self.sh_param * rcf
        elif sh_relation.lower() == 'const':
            length = np.ones(rcf.shape) * Disk.AU
            self.H = self.sh_param * length
        else:
            print 'WARNING::Could not determine scale height structure from given inputs. Please check sh_relation.'

    def density(self):
        'Return the density structure'
        return self.rho0

    def temperature(self):
        'Return the temperature structure'
        return self.tempg

    def dust_density(self):
        'Return the dust density structure'
        return self.rhoD0

    # def grid(self):
    #     'Return an XYZ grid (but which one??)'
    #     return self.grid

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
