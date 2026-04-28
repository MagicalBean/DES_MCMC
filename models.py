import numpy as np
from astropy.cosmology import LambdaCDM, FlatLambdaCDM, FlatwCDM, Flatw0waCDM

class BaseCosmologyModel:
    def __init__(self):
        self.param_names = []
        self.param_symbols = []
        self.ndim = 0
        
    def ln_prior(self, params):
        raise NotImplementedError
    
    def mu_model(self, params, zs):
        raise NotImplementedError
    
class LambdaCDMModel(BaseCosmologyModel):
    """Full LambdaCDM model"""
    def __init__(self):
        self.name = 'lCDM'
        self.param_names = ['Hubble Constant', 'Matter Density Parameter', 'Dark Energy Parameter']
        self.param_symbols = ["$H_0$", "$\Omega_m$", "$\Omega_\Lambda$"]
        self.ndim = 3

    def ln_prior(self, params):
        H_0, Om_m, Om_l = params
        
        if H_0 < 0:
            return -np.inf
        elif Om_m < 0 or Om_m > 1:
            return -np.inf
        elif Om_l < 0 or Om_l > 1:
            return -np.inf
        else:
            return 0
        
    def mu_model(self, params, zs):
        H_0, Om_m, Om_l = params
        cosmo = LambdaCDM(H0=H_0, Om0=Om_m, Ode0=Om_l) # use astropy for more acurate computing of the comoving distance
        return cosmo.distmod(zs).value
    
class FlatLambdaCDMModel(BaseCosmologyModel):
    """Flat Lambda CDM model (Omega_k = 0)"""
    def __init__(self):
        self.name = 'flCDM'
        self.param_names = ['Hubble Constant', 'Matter Density Parameter']
        self.param_symbols = ["$H_0$", "$\Omega_m$"]
        self.ndim = 2

    def ln_prior(self, params):
        H_0, Om_m = params
        
        if H_0 <= 0:
            return -np.inf
        if not (0 <= Om_m <= 1):
            return -np.inf
        
        return 0

    def mu_model(self, params, sz):
        H_0, Om_m = params
        cosmo = FlatLambdaCDM(H0=H_0, Om0=Om_m)
        return cosmo.distmod(sz).value
    
class FlatwCDMModel(BaseCosmologyModel):
    """Flat wCDM model (Omega_k = 0, w free)"""
    def __init__(self):
        self.name = 'FlatwCDM'
        self.param_names = ['Hubble Constant', 'Matter Density Parameter', 'w']
        self.param_symbols = ["$H_0$", "$\Omega_m$", "$\omega"]
        self.ndim = 3

    def ln_prior(self, params):
        H_0, Om_m, w = params
        
        if H_0 <= 0:
            return -np.inf
        if not (0 <= Om_m <= 1):
            return -np.inf
        
        # TODO: figure out priors
        
        return 0

    def mu_model(self, params, zs):
        H_0, Om_m, w = params
        
        cosmo = FlatwCDM(H0=H_0, Om0=Om_m, w0=w)
        return cosmo.distmod(zs).value

class Flatw0waCDMModel(BaseCosmologyModel):
    """Float w0waCDM Model"""
    def __init__(self):
        self.name = 'Flatw0waCDM'
        self.param_names = ['Hubble Constant', 'Matter Density Parameter', 'w0', 'wa']
        self.param_symbols = ["$H_0$", "$\Omega_m$", "$\omega_0$", "$\omega_a$"]
        self.ndim = 4

    def ln_prior(self, params):
        H_0, Om_m, w0, wa = params
        
        if H_0 <= 0:
            return -np.inf
        if not (0 <= Om_m <= 1):
            return -np.inf
        
        # TODO: figure out priors
                
        return 0

    def mu_model(self, params, zs):
        H_0, Om_m, w0, wa = params
        
        cosmo = Flatw0waCDM(H0=H_0, Om0=Om_m, w0=w0, wa=wa)
        return cosmo.distmod(zs).value

class MatterOnlyModel(BaseCosmologyModel):
    """"""

class EdSModel(BaseCosmologyModel):
    """Flat, matter-only universe, Om_m=1, Om_l=0, Om_k=0"""
    def __init__(self):
        self.name = 'EdS'
        self.param_names = ['Hubble Constant']
        self.param_symbols = ["$H_0$"]
        self.ndim = 1

    def ln_prior(self, params):
        H_0 = params[0]
        
        if H_0 < 0:
            return -np.inf
        else:
            return 0
        
    def mu_model(self, params, zs):
        H_0 = params[0]
        cosmo = LambdaCDM(H0=H_0, Om0=1.0, Ode0=0.0)
        return cosmo.distmod(zs).value