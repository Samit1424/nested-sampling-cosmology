"""
Nested Sampling, Explained by a Cosmologist -- companion code.

Reproduces the figures and evidence values in the Medium article:
  1. two-peaked toy likelihood (live points climbing both peaks)
  2. evidence accumulation, L * dX versus -ln X
  3. LCDM vs wCDM Bayes factor on 200 synthetic Type Ia supernovae

Run:  python nested_sampling_demo.py
Needs: numpy, scipy, matplotlib, dynesty
"""
import numpy as np, dynesty, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import quad
rng=np.random.default_rng(42)
plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False})

# ---- 1. bimodal toy: shrinking live points
def lnl_bi(x):
    a=-0.5*np.sum((x-np.array([-2.,-2.]))**2)/0.3**2
    b=-0.5*np.sum((x-np.array([2.5,1.5]))**2)/0.3**2
    return np.logaddexp(a,b)
def pt_bi(u): return 10*u-5
s=dynesty.NestedSampler(lnl_bi,pt_bi,2,nlive=300,rstate=rng)
s.run_nested(dlogz=0.05,print_progress=False)
r=s.results
smp=r.samples
fig,ax=plt.subplots(1,3,figsize=(12,3.8))
n=len(smp)
for a,(lo,hi,t) in zip(ax,[(0,300,"First 300 discarded points"),(n//3,n//3+300,"Middle of the run"),(n-600,n-300,"Near the end")]):
    a.scatter(smp[lo:hi,0],smp[lo:hi,1],s=6,c="#2b6cb0")
    a.set_xlim(-5,5);a.set_ylim(-5,5);a.set_title(t);a.set_aspect("equal")
plt.tight_layout();plt.savefig("fig1_shrinking.png",dpi=150);plt.close()
lnZ_true=np.log(2*2*np.pi*0.3**2/100)
print("bimodal lnZ",r.logz[-1],"+-",r.logzerr[-1],"true",lnZ_true,"ncall",r.ncall.sum())

# ---- 2. Evidence accumulation plot
logX=r.logvol[:-300]; logL=r.logl[:-300]; logwt=r.logwt[:-300]
fig,ax=plt.subplots(1,2,figsize=(11,3.8))
ax[0].plot(-logX,logL,c="#2b6cb0");ax[0].set_xlabel(r"$-\ln X$  (prior volume left, shrinking →)");ax[0].set_ylabel(r"$\ln L$");ax[0].set_ylim(logL.max()-25,logL.max()+2)
w=np.exp(logwt-logwt.max())
ax[1].plot(-logX,w/w.max(),c="#c05621");ax[1].set_xlabel(r"$-\ln X$");ax[1].set_ylabel(r"$L\,\Delta X$  (normalised)")
ax[1].set_title("Where the evidence actually lives")
plt.tight_layout();plt.savefig("fig2_evidence.png",dpi=150);plt.close()

# ---- 3. cosmology: synthetic SNe, LCDM vs wCDM
c=299792.458
def mu(z,Om,w,H0=70.):
    zz=np.atleast_1d(z);out=[]
    for zi in zz:
        I=quad(lambda x:1/np.sqrt(Om*(1+x)**3+(1-Om)*(1+x)**(3*(1+w))),0,zi)[0]
        out.append(5*np.log10((1+zi)*c/H0*I)+25)
    return np.array(out)
zgrid=np.linspace(0.005,1.6,400)
def mu_fast(Om,w):
    E=np.sqrt(Om*(1+zgrid)**3+(1-Om)*(1+zgrid)**(3*(1+w)))
    I=np.concatenate([[zgrid[0]/E[0]],zgrid[0]/E[0]+np.cumsum(0.5*(1/E[1:]+1/E[:-1])*np.diff(zgrid))])
    return 5*np.log10((1+zgrid)*c/70*I)+25
z=np.sort(rng.uniform(0.01,1.5,200)); sig=0.15
idx=np.searchsorted(zgrid,z)
truth=mu(z,0.3,-1.0)
data=truth+rng.normal(0,sig,len(z))
np.savetxt("synthetic_sne.txt",np.c_[z,data,np.full_like(z,sig)],header="z mu sigma")
def lnl_l(p):
    m=np.interp(z,zgrid,mu_fast(p[0],-1.0)); return -0.5*np.sum((data-m)**2)/sig**2
def lnl_w(p):
    m=np.interp(z,zgrid,mu_fast(p[0],p[1])); return -0.5*np.sum((data-m)**2)/sig**2
pt_l=lambda u: np.array([0.05+0.65*u[0]])
pt_w=lambda u: np.array([0.05+0.65*u[0],-2.0+1.5*u[1]])
out={}
for name,f,pt,nd in [("LCDM",lnl_l,pt_l,1),("wCDM",lnl_w,pt_w,2)]:
    ss=dynesty.NestedSampler(f,pt,nd,nlive=400,rstate=np.random.default_rng(1))
    ss.run_nested(dlogz=0.05,print_progress=False)
    rr=ss.results; out[name]=rr
    wts=np.exp(rr.logwt-rr.logz[-1])
    means=np.average(rr.samples,axis=0,weights=wts)
    print(name,"lnZ",rr.logz[-1],"+-",rr.logzerr[-1],"maxlnL",rr.logl.max(),"means",means,"ncall",rr.ncall.sum())
lnB=out["LCDM"].logz[-1]-out["wCDM"].logz[-1]
print("lnB LCDM vs wCDM",lnB, "B=",np.exp(lnB))
print("dmaxlnL w-l",out["wCDM"].logl.max()-out["LCDM"].logl.max())
# posterior plot wCDM
rr=out["wCDM"];wts=np.exp(rr.logwt-rr.logz[-1]);
from dynesty import utils as dyfunc
eq=dyfunc.resample_equal(rr.samples,wts/wts.sum())
fig,ax=plt.subplots(1,2,figsize=(11,4))
ax[0].errorbar(z,data,sig,fmt=".",ms=3,c="#718096",alpha=.6,label="synthetic SNe")
ax[0].plot(zgrid,mu_fast(0.3,-1),c="#2b6cb0",label="ΛCDM truth");ax[0].set_xlabel("z");ax[0].set_ylabel("distance modulus μ");ax[0].legend()
ax[1].hist2d(eq[:,0],eq[:,1],bins=50,cmap="Blues");ax[1].axhline(-1,c="#c05621",ls="--");ax[1].set_xlabel(r"$\Omega_m$");ax[1].set_ylabel("w");ax[1].set_title("wCDM posterior (from the same run)")
plt.tight_layout();plt.savefig("fig3_cosmo.png",dpi=150);plt.close()
print("w 68%",np.percentile(eq[:,1],[16,50,84]),"Om",np.percentile(eq[:,0],[16,50,84]))
