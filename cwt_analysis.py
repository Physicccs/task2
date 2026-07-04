import pandas as pd, numpy as np

# ---- Torrence & Compo FFT-based Morlet CWT ----
W0 = 6.0  # Morlet omega0

def morlet_cwt(x, dt, periods):
    n = len(x)
    npad = int(2**np.ceil(np.log2(n)))
    xf = np.fft.fft(x, npad)
    omega = 2*np.pi*np.fft.fftfreq(npad, dt)
    scales = periods * (W0 + np.sqrt(2+W0**2)) / (4*np.pi)  # Fourier factor
    coef = np.empty((len(scales), n), dtype=np.complex64)
    for i, s in enumerate(scales):
        psi = np.zeros(npad)
        pos = omega > 0
        psi[pos] = (np.pi**-0.25) * np.exp(-0.5*(s*omega[pos]-W0)**2) * np.sqrt(2*np.pi*s/dt)
        coef[i] = np.fft.ifft(xf*psi)[:n]
    return coef

df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
df.columns = ['t','pos','val']
T = int(df.t.max())+1
x = np.zeros(T)
for t in df.t.values: x[t]+=1
x -= x.mean()

periods = np.logspace(np.log10(2), np.log10(3600), 120)
coef = morlet_cwt(x, 1.0, periods)
power = np.abs(coef)**2
gws = power.mean(axis=1)
print("cwt done", power.shape)

rng = np.random.default_rng(42)
ts_sorted = np.sort(df.t.values)
gaps = np.diff(ts_sorted)

def gws_of(tt):
    xs = np.zeros(T)
    for t in tt: xs[min(int(t),T-1)]+=1
    xs -= xs.mean()
    c = morlet_cwt(xs, 1.0, periods)
    return (np.abs(c)**2).mean(axis=1)

# surrogate A: shuffled gaps (keeps gap distribution, kills ordering/periodicity)
n_sur = 200
gws_sur = np.zeros((n_sur, len(periods)))
for i in range(n_sur):
    g = rng.permutation(gaps)
    tt = np.concatenate([[ts_sorted[0]], ts_sorted[0]+np.cumsum(g)])
    gws_sur[i] = gws_of(tt)
sig95 = np.percentile(gws_sur,95,axis=0); sig99 = np.percentile(gws_sur,99,axis=0)
print("shuffle-surrogate periods exceeding 95%:", np.round(periods[gws>sig95],1))
print("shuffle-surrogate periods exceeding 99%:", np.round(periods[gws>sig99],1))

# surrogate B: homogeneous Poisson with same rate
n_poi = 200
gws_poi = np.zeros((n_poi, len(periods)))
lam = len(df)/T
for i in range(n_poi):
    xs = rng.poisson(lam, T).astype(float); xs -= xs.mean()
    c = morlet_cwt(xs, 1.0, periods)
    gws_poi[i] = (np.abs(c)**2).mean(axis=1)
poi95 = np.percentile(gws_poi,95,axis=0); poi05 = np.percentile(gws_poi,5,axis=0)
above = periods[gws>poi95]; below = periods[gws<poi05]
print("\nvs Poisson: periods ABOVE 95%:", np.round(above,1))
print("vs Poisson: periods BELOW 5%:", np.round(below,1))

# cone of influence (e-folding time for Morlet, T&C): coi = sqrt(2)*scale
scales = periods*(W0+np.sqrt(2+W0**2))/(4*np.pi)
np.savez('cwt_results.npz', power=power.astype(np.float32), periods=periods,
         gws=gws, sig95=sig95, sig99=sig99, poi95=poi95, poi05=poi05,
         sur_mean=gws_sur.mean(axis=0), poi_mean=gws_poi.mean(axis=0),
         scales=scales, T=T)
print("saved")
