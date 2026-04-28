import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import emcee
from multiprocessing import Pool

class CosmologyMCMC:
    def __init__(self, model, data_df):
        self.model = model
        self.data_df = data_df
        self.burned = None
        
    def ln_likelihood(self, params):
        residuals = self.data_df['MU'] - self.model.mu_model(params, self.data_df['zHD'])    
        
        inv_sigma2 = 1.0 / self.data_df['MUERR']**2
        
        chit2 = np.sum(residuals**2 * inv_sigma2)
        B = np.sum(residuals * inv_sigma2)
        C = np.sum(inv_sigma2)
        
        chi2 = chit2 - (B**2 / C) + np.log(C / (2 * np.pi))
        return -0.5 * chi2

    def ln_pdf(self, params):
        lp = self.model.ln_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.ln_likelihood(params)
    
    def run_mcmc(self, nwalkers=16, nsteps=5000, p0=None):
        if p0 is None:
            p0 = np.random.rand(nwalkers, self.model.ndim)

        with Pool() as pool:
            start = time.time()
            self.sampler = emcee.EnsembleSampler(nwalkers, self.model.ndim, self.ln_pdf, pool=pool)
            self.sampler.run_mcmc(p0, nsteps, progress=True)
            dur = time.time() - start
            print(f'{self.model.name} model took {dur:.3f} seconds')

    def get_samples(self, discard=0, flat=False):
        return self.sampler.get_chain(discard=discard, flat=flat)
        
    def test_priors(self, p0):
        print([self.model.ln_prior(x) for x in p0])
    
    def compute_best_M(self, params):
        mu_model = self.model.mu_model(params, self.data_df['zHD'])
        residuals = self.data_df['MU'] - mu_model
        
        inv_sigma2 = 1.0 / self.data_df['MUERR']**2
        
        B = np.sum(residuals * inv_sigma2)
        C = np.sum(inv_sigma2)
        
        return B / C
    
    def mu_model_corrected(self, params, zs):
        mu = self.model.mu_model(params, zs)
        
        M_best = self.compute_best_M(params)
        
        return mu + M_best
        
    def trace_plots(self, burn_in=None):
        fig, axs = plt.subplots(self.model.ndim, 1, figsize=(12, 2*self.model.ndim))
        axs = np.atleast_1d(axs).flat
        
        for i in range(0, self.model.ndim):
            axs[i].plot(self.get_samples()[:,:,i], 'k', alpha=0.3)
            axs[i].set_title(self.model.param_names[i], size=16)
            axs[i].set_ylabel(self.model.param_symbols[i], size=14)
            if burn_in is not None:
                axs[i].axvline(x=burn_in, color='red', linestyle='dashed')

        plt.xlabel('Iterations', size=14)
        fig.suptitle('Trace Plots', size=18)
        plt.tight_layout()
        plt.show()
