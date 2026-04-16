import numpy as np
import json, sys

# ── Parameters ────────────────────────────────────────────────────────────
N  = 16      # array elements per dimension
sp = 0.5    # element spacing in wavelengths
L  = 10      # spatial smoothing subarray size (rule of thumb: ~2N/3)


def simulator(path,aoa,N=8, sp=0.5):

    ar = np.zeros(N*N, dtype=complex)
    for p, angles in zip(path, aoa):
        az  = np.deg2rad(angles[0])
        el  = np.deg2rad(angles[1])
        ux  = np.sin(az) * np.cos(el)
        uy  = np.sin(az) * np.sin(el)
        m   = np.arange(N)
        sv1 = np.exp(1j*2*np.pi*sp * m * ux)
        sv2 = np.exp(1j*2*np.pi*sp * m * uy)
        ar += p * np.outer(sv1, sv2).flatten()
    return ar, path, aoa   


def steering_vector(N, sp, az_deg, el_deg):
    az  = np.deg2rad(az_deg)
    el  = np.deg2rad(el_deg)
    ux  = np.sin(az) * np.cos(el)
    uy  = np.sin(az) * np.sin(el)
    m   = np.arange(N)
    sv1 = np.exp(1j*2*np.pi*sp * m * ux)
    sv2 = np.exp(1j*2*np.pi*sp * m * uy)
    return np.outer(sv1, sv2).flatten()        # (N*N,)

def spatial_smooth_2d(ar, N, L):
    ar_2d = ar.reshape(N, N)
    K     = N - L + 1
    R_ss  = np.zeros((L*L, L*L), dtype=complex)
    for i in range(K):
        for j in range(K):
            sub   = ar_2d[i:i+L, j:j+L].flatten()
            R_ss += np.outer(sub, sub.conj())
    return R_ss / (K*K)

def capon_power(R_inv, L, sp, az_grid, el_grid):
    AZ, EL   = np.meshgrid(np.deg2rad(az_grid), np.deg2rad(el_grid))
    ux       = np.sin(AZ) * np.cos(EL)
    uy       = np.sin(AZ) * np.sin(EL)
    m        = np.arange(L)
    sm1      = np.exp(1j*2*np.pi*sp * m[:, None, None] * ux[None, :, :])
    sm2      = np.exp(1j*2*np.pi*sp * m[:, None, None] * uy[None, :, :])
    sm       = (sm1.reshape(L,-1)[:, None, :] *
                sm2.reshape(L,-1)[None, :, :]).reshape(L*L, -1)
    denom    = np.real(np.sum(sm.conj() * (R_inv @ sm), axis=0))
    return (1.0 / (denom + 1e-15)).reshape(len(el_grid), len(az_grid))

def extract_fingerprint(ar, N, sp, L,
                        coarse_step=5, fine_step=0.5, fine_window=10):
    """
    Compresses antenna response ar (N*N,) into [az, el, mag] 
    of the strongest path. Phase excluded for real-world robustness.
    Magnitude normalised against total received power.
    """
    # Covariance
    R     = spatial_smooth_2d(ar, N, L)
    R    += np.eye(L*L) * 1e-2
    R_inv = np.linalg.inv(R)

    # Coarse scan
    az_c     = np.arange(-90, 91, coarse_step)
    el_c     = np.arange(-90, 91, coarse_step)
    P_c      = capon_power(R_inv, L, sp, az_c, el_c)
    ei, ai   = np.unravel_index(np.argmax(P_c), P_c.shape)
    az0, el0 = az_c[ai], el_c[ei]

    # Fine scan around coarse peak
    az_f   = np.clip(np.arange(az0-fine_window, az0+fine_window, fine_step), -90, 90)
    el_f   = np.clip(np.arange(el0-fine_window, el0+fine_window, fine_step), -90, 90)
    P_f    = capon_power(R_inv, L, sp, az_f, el_f)
    ei, ai = np.unravel_index(np.argmax(P_f), P_f.shape)
    az_est = az_f[ai]
    el_est = el_f[ei]

    # MVDR amplitude at peak — magnitude only
    a      = steering_vector(L, sp, az_est, el_est)
    Rinv_a = R_inv @ a
    w      = Rinv_a / (a.conj() @ Rinv_a)
    mag    = np.abs(w.conj() @ ar[:L*L])

    # Normalise magnitude against total received power
    mag_norm = mag / np.sqrt(np.mean(np.abs(ar)**2))

    return np.array([az_est, el_est, mag_norm])

# ── Test ──────────────────────────────────────────────────────────────────

bts=[[],[],[],[],[],[],[],[],[]]
with open('./MLWC/small_otaniemi_matlab_mat.json','r') as ff: data=np.array(json.load(ff))
ants=[-75,25,	125,	100,	125,	-150,	-150,	100,	150] #antenna azimuth rotation
nbts=len(ants); nue=len(data)

outv=np.zeros((nue,nbts,3))
ii=-1;
for kk in range(nbts*nue-nue):
  if kk%9==0:ii+=1
  dd=data[ii,kk%9]
  
  if sum(dd[0])==0: continue
  
  if ii==nbts: ii+=1
  path=dd[0]+1j*dd[1]; aoa=np.array([dd[2]-ants[kk%9],dd[3]]).T
  
  ar, path, aoa = simulator(path, aoa,N)

  # Strongest true path for validation
  strongest_idx = np.argmax(np.abs(path))
  az_true       = aoa[strongest_idx, 0]
  el_true       = aoa[strongest_idx, 1]
  mag_true      = np.abs(path[strongest_idx])
  
  #bts[kk%9].append(az_true)
  
  fp = extract_fingerprint(ar, N, sp, L)
  outv[ii,kk%nbts]=[fp[0],fp[1],fp[2]]

  #print(f"{'':20} {'Az (°)':>8}  {'El (°)':>8}  {'|mag| norm':>10}")
  #print("-" * 50)
  #print(f"{'True (strongest)':20} {az_true:>8.1f}  {el_true:>8.1f}  {mag_true/np.sqrt(np.mean(np.abs(ar)**2)):>10.4f}")
  #print(f"{'Estimated':20} {fp[0]:>8.1f}  {fp[1]:>8.1f}  {fp[2]:>10.4f}")
  #print(f"{'Error':20} {abs(fp[0]-az_true):>8.1f}  {abs(fp[1]-el_true):>8.1f}  {abs(fp[2]-mag_true/np.sqrt(np.mean(np.abs(ar)**2))):>10.4f}")
  #print()

with open('aod_otaniemi_16b16.json','w') as ff: json.dump(outv.tolist(),ff,indent=2)
sys.exit()
for bb in bts:
  h,b=np.histogram(bb,np.arange(-180,180,5))
  for hh in h: print(hh,end=' ')
  print()