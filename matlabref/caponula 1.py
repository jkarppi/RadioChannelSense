import numpy as np
import json, sys

# Parameters
dims = 2    # antenna dimensions
N  = 4      # array elements per dimension
sp = 0.5    # element spacing in wavelengths
L  = 3      # spatial smoothing (rule of thumb: 2D ~2N/3, 1D N//2)

with open('./MLWC/small_otaniemi_matlab_mat.json','r') as ff: data=np.array(json.load(ff)) # ue_count x bts_count x [real x 16,imag x 16,azimuth x 16,elevation x 16,delay x 16], zeros(5,16) if no bts
timingdata=True #create timing data file only
ants=[-75,25,	125,	100,	125,	-150,	-150,	100,	150] #antenna azimuth rotation

fname='aod_otaniemi_4b4.json'
#fname='timing_otaniemi.json'


def calc_log_euc(m):
  
  M=np.zeros((len(m),len(m)))
  for ii,mm in enumerate(m):
    M[ii]=np.sqrt(np.sum(np.sum(np.abs(m-mm)**2,1),1))
  return M.tolist()
  
def antenna(path,aoa,N=8, dims=2,sp=0.5):

    if dims == 1:
        ar = np.zeros(N, dtype=complex)
        for p, angles in zip(path, aoa):
            ar += p * steering_vector(N, sp, angles[0], angles[1], 1)
    else:
        ar = np.zeros(N*N, dtype=complex)
        for p, angles in zip(path, aoa):
            ar += p * steering_vector(N, sp, angles[0], angles[1], 2)
    return ar, path, aoa   


def steering_vector(N, sp, az_deg, el_deg,dims=2):
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    m  = np.arange(N)
    if dims==1:
        # ULA — azimuth only, elevation projects onto array axis
        ux = np.sin(az) * np.cos(el)
        return np.exp(1j*2*np.pi*sp * m * ux)             # (N,)
    else:
        # URA — both azimuth and elevation
        ux  = np.sin(az) * np.cos(el)
        uy  = np.sin(az) * np.sin(el)
        sv1 = np.exp(1j*2*np.pi*sp * m * ux)
        sv2 = np.exp(1j*2*np.pi*sp * m * uy)
        return np.outer(sv1, sv2).flatten()                # (N*N,)

def spatial_smooth_2d(ar, N, L,dims=2):
    if dims==1:
        # 1D sliding window
        K    = N - L + 1
        R_ss = np.zeros((L, L), dtype=complex)
        for i in range(K):
            sub   = ar[i:i+L]
            R_ss += np.outer(sub, sub.conj())
        return R_ss / K
    else:
        # 2D sliding window
        ar_2d = ar.reshape(N, N)
        K     = N - L + 1
        R_ss  = np.zeros((L*L, L*L), dtype=complex)
        for i in range(K):
            for j in range(K):
                sub   = ar_2d[i:i+L, j:j+L].flatten()
                R_ss += np.outer(sub, sub.conj())
        return R_ss / (K*K)

def capon_power(R_inv, L, sp, az_grid, el_grid,dims=2):
    m = np.arange(L)
    if dims==1:
        # Scan azimuth only — elevation folds into ux projection
        # el_grid ignored for ULA but kept as argument for consistent interface
        P = np.zeros((1, len(az_grid)))
        for ai, az in enumerate(az_grid):
            a       = steering_vector(L, sp, az, 0, 1)
            Rinv_a  = R_inv @ a
            denom   = np.real(a.conj() @ Rinv_a)
            P[0,ai] = 1.0 / (denom + 1e-15)
        return P   # (1, n_az) — consistent shape with 2D case
    else:
        AZ, EL   = np.meshgrid(np.deg2rad(az_grid), np.deg2rad(el_grid))
        ux       = np.sin(AZ) * np.cos(EL)
        uy       = np.sin(AZ) * np.sin(EL)
        sm1      = np.exp(1j*2*np.pi*sp * m[:, None, None] * ux[None, :, :])
        sm2      = np.exp(1j*2*np.pi*sp * m[:, None, None] * uy[None, :, :])
        sm       = (sm1.reshape(L,-1)[:, None, :] *
                    sm2.reshape(L,-1)[None, :, :]).reshape(L*L, -1)
        denom    = np.real(np.sum(sm.conj() * (R_inv @ sm), axis=0))
        return (1.0 / (denom + 1e-15)).reshape(len(el_grid), len(az_grid))

def extract_fingerprint(ar, N, sp, L,dims=2,
                        coarse_step=5, fine_step=0.5, fine_window=10):
    """
    Returns [az, mag_norm]        for ULA (1D)
    Returns [az, el, mag_norm]    for URA (2D)
    Phase excluded for real-world robustness.
    Magnitude normalised against total received power.
    """
    R     = spatial_smooth_2d(ar, N, L, dims)
    R    += np.eye(R.shape[0]) * 1e-2
    R_inv = np.linalg.inv(R)

    az_c = np.arange(-90, 91, coarse_step)
    el_c = np.arange(-90, 91, coarse_step) if dims==2 else np.array([0.0])

    # Coarse scan
    P_c      = capon_power(R_inv, L, sp, az_c, el_c, dims)
    ei, ai   = np.unravel_index(np.argmax(P_c), P_c.shape)
    az0      = az_c[ai]
    el0      = el_c[ei] if dims==2 else 0.0

    # Fine scan
    az_f   = np.clip(np.arange(az0-fine_window, az0+fine_window, fine_step), -90, 90)
    el_f   = (np.clip(np.arange(el0-fine_window, el0+fine_window, fine_step), -90, 90)
              if dims==2 else np.array([0.0]))
    P_f    = capon_power(R_inv, L, sp, az_f, el_f, dims)
    ei, ai = np.unravel_index(np.argmax(P_f), P_f.shape)
    az_est = az_f[ai]
    el_est = el_f[ei] if dims==2 else 0.0

    # MVDR amplitude
    sv_len = L if dims==2 else L*L
    a      = steering_vector(L, sp, az_est, el_est, dims)
    Rinv_a = R_inv @ a
    w      = Rinv_a / (a.conj() @ Rinv_a)
    mag    = np.abs(w.conj() @ ar[:sv_len])

    mag_norm = mag / np.sqrt(np.mean(np.abs(ar)**2))

    if dims==1:
        return np.array([az_est, mag_norm])            # 2 features
    else:
        return np.array([az_est, el_est, mag_norm])    # 3 features



#run here
bts=[[],[],[],[],[],[],[],[],[]]
nbts=len(ants); nue=len(data)

outv=np.zeros((nue,nbts,3))
ii=-1;
for kk in range(nbts*nue-nue):
  if kk%nbts==0:ii+=1
  dd=data[ii,kk%nbts]
  
  if sum(dd[0])==0: continue #no bts
  
  if ii==nbts: ii+=1
  path=dd[0]+1j*dd[1]; aoa=np.array([dd[2]-ants[kk%nbts],dd[3]]).T
  
  ar, path, aoa = antenna(path, aoa,N)
  
  dls=np.array(dd[3])-min(dd[3])
  ps=np.abs(path)**2
  mds=sum(dls*ps)/sum(ps) #mean delay spread
  mdss=sum((dls**2)*ps)/sum(ps) 
  vds=np.sqrt(mdss-mds**2) #delay spread variance
  #md=max(np.abs(p))
  midx=np.argmax(ps[np.argsort(dls)])  #strong path idx
  if timingdata:
    #bts[kk%nbts].append(az_true)
    bts[kk%nbts].append(mds)
    outv[ii,kk%nbts]=[mds,vds,midx]
    continue

  # Strongest true path for validation
  strongest_idx = np.argmax(np.abs(path))
  az_true       = aoa[strongest_idx, 0]
  el_true       = aoa[strongest_idx, 1]
  mag_true      = np.abs(path[strongest_idx])
  
  fp = extract_fingerprint(ar, N, sp, L)
  outv[ii,kk%nbts]=[fp[0],fp[1],fp[2]]

  #print(f"{'':20} {'Az (°)':>8}  {'El (°)':>8}  {'|mag| norm':>10}")
  #print("-" * 50)
  #print(f"{'True (strongest)':20} {az_true:>8.1f}  {el_true:>8.1f}  {mag_true/np.sqrt(np.mean(np.abs(ar)**2)):>10.4f}")
  #print(f"{'Estimated':20} {fp[0]:>8.1f}  {fp[1]:>8.1f}  {fp[2]:>10.4f}")
  #print(f"{'Error':20} {abs(fp[0]-az_true):>8.1f}  {abs(fp[1]-el_true):>8.1f}  {abs(fp[2]-mag_true/np.sqrt(np.mean(np.abs(ar)**2))):>10.4f}")
  #print()

with open(fname,'w') as ff: json.dump(outv.tolist(),ff,indent=2)

if not timingdata: sys.exit()
for bb in bts:
  h,b=np.histogram(bb)
  for ii in range(len(h)): print((b[ii]+b[ii+1])/2,h[ii])
  print()