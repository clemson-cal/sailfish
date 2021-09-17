import sys
import numpy as np
import msgpack
import re

def reconstitute(filename, fieldnum):
    chkpt = msgpack.load(open(filename, 'rb'))
    mesh = chkpt[b'mesh']
    primitive = np.zeros([mesh[b'ni'], mesh[b'nj'], 4])
    for patch in chkpt[b'primitive_patches']:
        i0 = patch[b'rect'][0][b'start']
        j0 = patch[b'rect'][1][b'start']
        i1 = patch[b'rect'][0][b'end']
        j1 = patch[b'rect'][1][b'end']
        local_prim = np.array(np.frombuffer(patch[b'data'])).reshape([i1 - i0, j1 - j0, 4])
        primitive[i0:i1, j0:j1] = local_prim
    rho, vx, vy, pres = primitive[:,:,0], primitive[:,:,1], primitive[:,:,2], primitive[:,:,3]
    if fieldnum==-1:
        return rho, vx, vy, pres
    if fieldnum==0:
        return rho
    if fieldnum==1:
        return vx
    if fieldnum==2:
        return vy
    if fieldnum==3:
        return pres
    par = re.search('gamma_law_index=(.+?):', chkpt[b'parameters'].decode())
    if par==None:
        gamma = 1.666666666666666
    else:
        gamma = np.float(par.group(1))
    if fieldnum==4:
        return pres / rho / (gamma - 1.)
    if fieldnum==5:
        cs = np.sqrt(gamma*pres/rho)
        v  = np.sqrt(vx**2 + vy**2)
        return v/cs

nstr       = str(np.char.zfill(str(Nchkpts[0]),4))
d          = msgpack.load(open(fn+'chkpt.'+nstr+'.sf', 'rb'))
print(d[b'parameters'])
DR         = np.float(re.search('domain_radius=(.+?):', d[b'parameters'].decode()).group(1))
N          = d[b'mesh'][b'ni']
dx         = d[b'mesh'][b'dx']
dy         = dx*1
cfl        = d[b'command_line'][b'cfl_number']
gamma      = np.float(re.search('gamma_law_index=(.+?):', d[b'parameters'].decode()).group(1))
coolcoef   = np.float(re.search('cooling_coefficient=(.+?):', d[b'parameters'].decode()).group(1))
buffrate   = 10.0
buffwidth  = 0.1
sinkrad    = np.float(re.search('sink_radius=(.+?):', d[b'parameters'].decode()).group(1))
sinkrate   = np.float(re.search('sink_rate=(.+?):', d[b'parameters'].decode()).group(1))
m1,m2      = d[b'masses'][0][b'mass'], d[b'masses'][1][b'mass']
alpha      = np.float(re.search('alpha=(.+?):', d[b'parameters'].decode()).group(1))

x          = np.arange((N))*dx - 2*DR/2. + dx/2.
xx,yy      = np.zeros((N,N)),np.zeros((N,N))
for i in range(N):
    xx[:,i] = x*1
    yy[i,:] = x*1
rr        = np.sqrt(xx**2+yy**2)

buffer_mask = np.ones((N,N)) #for summing errors only where buffer is inactive
for i in range(N):
    for j in range(N):
        if rr[i,j]>DR-buffwidth:
            buffer_mask[i,j] = 0.0

def dery(v,dy):
    dv = v*0
    dv[:,1:-1] = v[:,2:] - v[:,:-2]
    dv[:,0]    = 0.0
    dv[:,  -1] = 0.0
    return dv / (2.*dy)

def derx(v,dx):
    dv = v*0
    dv[1:-1,:] = v[2:,:] - v[:-2,:]
    dv[0,   :] = 0.0
    dv[-1,  :] = 0.0
    return dv / (2.*dx)

def dert(v1,v2,t1,t2):
    return (v2-v1)/(t2-t1)

def momentum(rho,v):
    return rho*v

def energy(rho,vx,vy,eps):
    return rho*eps + 0.5*rho*(vx**2+vy**2)

def compute_dert( rho_nm1, rho_np1,\
                   vx_nm1,  vx_np1,\
                   vy_nm1,  vy_np1,\
                  eps_nm1, eps_np1,\
                    t_nm1,   t_np1):
    dt_rho_n  = dert(          rho_nm1         ,          rho_np1         , t_nm1, t_np1 )
    dt_momx_n = dert( momentum(rho_nm1, vx_nm1), momentum(rho_np1, vx_np1), t_nm1, t_np1 )
    dt_momy_n = dert( momentum(rho_nm1, vy_nm1), momentum(rho_np1, vy_np1), t_nm1, t_np1 )
    dt_en_n   = dert( energy(rho_nm1, vx_nm1, vy_nm1, eps_nm1), \
                      energy(rho_np1, vx_np1, vy_np1, eps_np1), t_nm1, t_np1 )
    return dt_rho_n, dt_momx_n, dt_momy_n, dt_en_n

def inviscid_fluxes_for_derx( rho, vx, vy, pres, eps ):
    rhoflux  = rho*vx
    momxflux = rho*vx*vx + pres
    momyflux = rho*vy*vx
    enflux   = (rho*eps + 0.5*rho*(vx**2 + vy**2) + pres)*vx
    return rhoflux, momxflux, momyflux, enflux

def inviscid_fluxes_for_dery( rho, vx, vy, pres, eps ):
    rhoflux  = rho*vy
    momxflux = rho*vx*vy
    momyflux = rho*vy*vy + pres
    enflux   = (rho*eps + 0.5*rho*(vx**2 + vy**2) + pres)*vy
    return rhoflux, momxflux, momyflux, enflux

def viscid_fluxes_for_derx( rho, vx, vy, pres, eps, nu ):
    lam      = 0
    dxvx     = derx(vx,dx)
    dyvy     = dery(vy,dy)
    dxvy     = derx(vy,dx)
    dyvx     = dery(vx,dy)
    div      = dxvx + dyvy
    tau_xx   = rho * nu * ( 2*dxvx - (2./3)*div ) + rho * lam * div
    tau_yy   = rho * nu * ( 2*dyvy - (2./3)*div ) + rho * lam * div
    tau_xy   = rho * nu * ( dyvx + dxvy )
    tau_yx   = tau_xy*1
    momxflux = tau_xx
    momyflux = tau_xy
    enflux   = vx*tau_xx + vy*tau_xy
    return momxflux, momyflux, enflux

def viscid_fluxes_for_dery( rho, vx, vy, pres, eps, nu ):
    lam      = 0
    dxvx     = derx(vx,dx)
    dyvy     = dery(vy,dy)
    dxvy     = derx(vy,dx)
    dyvx     = dery(vx,dy)
    div      = dxvx + dyvy
    tau_xx   = rho * nu * ( 2*dxvx - (2./3)*div ) + rho * lam * div
    tau_yy   = rho * nu * ( 2*dyvy - (2./3)*div ) + rho * lam * div
    tau_xy   = rho * nu * ( dyvx + dxvy )
    tau_yx   = tau_xy*1
    momxflux = tau_yx
    momyflux = tau_yy
    enflux   = vx*tau_yx + vy*tau_yy
    return momxflux, momyflux, enflux

def kinematic_viscosity( rho, pres, x, y, x1, y1, x2, y2, m1, m2):
    cs2   = gamma*pres/rho
    r1    = np.sqrt((x-x1)**2 + (y-y1)**2)
    r2    = np.sqrt((x-x2)**2 + (y-y2)**2)
    twof  = 1./np.sqrt(m1/r1**3 + m2/r2**3)
    return alpha*cs2*twof/np.sqrt(gamma)

def compute_all_fluxes_der( rho, vx, vy, pres, eps, x, y, x1, y1, x2, y2, m1, m2 ):
    nu = kinematic_viscosity(rho,pres,x,y,x1,y1,x2,y2,m1,m2)
    rhoflux_invisc_x, momxflux_invisc_x, momyflux_invisc_x, enflux_invisc_x = inviscid_fluxes_for_derx( rho, vx, vy, pres, eps )
    rhoflux_invisc_y, momxflux_invisc_y, momyflux_invisc_y, enflux_invisc_y = inviscid_fluxes_for_dery( rho, vx, vy, pres, eps )
    momxflux_visc_x, momyflux_visc_x, enflux_visc_x = viscid_fluxes_for_derx( rho, vx, vy, pres, eps, nu )
    momxflux_visc_y, momyflux_visc_y, enflux_visc_y = viscid_fluxes_for_dery( rho, vx, vy, pres, eps, nu )
    d_rhoflux  = derx( rhoflux_invisc_x,dx) + dery( rhoflux_invisc_y,dy)
    d_momxflux = derx(momxflux_invisc_x,dx) + dery(momxflux_invisc_y,dy) - derx(momxflux_visc_x,dx) - dery(momxflux_visc_y,dy)
    d_momyflux = derx(momyflux_invisc_x,dx) + dery(momyflux_invisc_y,dy) - derx(momyflux_visc_x,dx) - dery(momyflux_visc_y,dy)
    d_enflux   = derx(  enflux_invisc_x,dx) + dery(  enflux_invisc_y,dy) - derx(  enflux_visc_x,dx) - dery(  enflux_visc_y,dy)
    return d_rhoflux, d_momxflux, d_momyflux, d_enflux

def disk_height(rho, pres, x, y, x1, y1, x2, y2):
    r1   = np.sqrt((x-x1)**2 + (y-y1)**2 + 1e-12)
    r2   = np.sqrt((x-x2)**2 + (y-y2)**2 + 1e-12)
    omeg = np.sqrt(m1/r1**3 + m2/r2**3)
    return np.sqrt(pres/rho)/omeg

def sources_gravity( rho, pres, vx, vy, x, y, x1, y1, x2, y2 ):
    r1   = np.sqrt((x-x1)**2 + (y-y1)**2)
    r2   = np.sqrt((x-x2)**2 + (y-y2)**2)
    h    = disk_height(rho, pres, x, y, x1, y1, x2, y2)
    rs1  = 0.5 * h
    rs2  = 0.5 * h
    for i in range(np.shape(r1)[0]):
        for j in range(np.shape(r1)[1]):
            if r1[i,j]<sinkrad:
                transition = (1.0 - (r1[i,j]/sinkrad)**2)**2
                rs1[i,j]   = transition*sinkrad + (1-transition)*0.5*h[i,j]
            if r2[i,j]<sinkrad:
                transition = (1.0 - (r2[i,j]/sinkrad)**2)**2
                rs2[i,j]   = transition*sinkrad + (1-transition)*0.5*h[i,j]
    fpre1= -rho * m1/(r1**2 + rs1**2)**(3./2)
    fpre2= -rho * m2/(r2**2 + rs2**2)**(3./2)
    f1x  = fpre1 * (x-x1)
    f1y  = fpre1 * (y-y1)
    f2x  = fpre2 * (x-x2)
    f2y  = fpre2 * (y-y2)
    momxsrc = f1x + f2x
    momysrc = f1y + f2y
    ensrc   = vx*f1x + vy*f1y + vx*f2x + vy*f2y
    return momxsrc, momysrc, ensrc

def sources_buffer( rho, vx, vy, pres, eps, rho0, vx0, vy0, pres0, eps0, x, y ): #inactive in testing mode
    r = np.sqrt(x**2 + y**2)
    massbuff = (rho - 1.0)
    momxbuff = (rho*vx)
    momybuff = (rho*vy)
    v2       = vx**2  + vy**2
    v20      = vx0**2 + vy0**2
    en       = rho *eps  + 0.5*rho *v2
    en0      = rho0*eps0 + 0.5*rho0*v20
    enbuff   = (en - 1.0/(5./3-1))
    rbuff    = r-(DR-buffwidth)
    omeg_out = 0.0
    buffwind = buffrate * omeg_out * (r - rbuff) / (DR - rbuff)
    massbuff*= -buffwind
    momxbuff*= -buffwind
    momybuff*= -buffwind
    enbuff  *= -buffwind
    return massbuff, momxbuff, momybuff, enbuff

def sources_cooling( rho, eps ):
	return -coolcoef * eps**4 / rho

def sources_sinks( rho, vx, vy, pres, eps, x, y, xbh, ybh, vxbh, vybh ): #torque-free
    rbh2   = (x-xbh)**2 + (y-ybh)**2
    rbh    = np.sqrt(rbh2)
    xhat   = (x-xbh)/(rbh+1e-12)
    yhat   = (y-ybh)/(rbh+1e-12)
    vdotr  = (vx-vxbh)*xhat + (vy-vybh)*yhat
    vxstar = vdotr*xhat + vxbh
    vystar = vdotr*yhat + vybh
    momx = momentum(rho,vxstar)
    momy = momentum(rho,vystar)
    en   = energy(rho,vxstar,vystar,eps)
    s2   = sinkrad**2
    sinkwind = np.exp(-(rbh2/s2)**2) * sinkrate
    for i in range(N):
        for j in range(N):
            if rbh[i,j] >= sinkrad * 4.0:
                sinkwind[i,j] = 0.0
    mdot    = -sinkwind * rho
    return -sinkwind * rho, -sinkwind * momx, -sinkwind * momy, -sinkwind * en


n       = Nchkpts[0]
nstr    = str(np.char.zfill(str(n),4))
rho_nm1 = reconstitute(fn+'chkpt.'+nstr+'.sf',0)
vx_nm1  = reconstitute(fn+'chkpt.'+nstr+'.sf',1)
vy_nm1  = reconstitute(fn+'chkpt.'+nstr+'.sf',2)
pres_nm1= reconstitute(fn+'chkpt.'+nstr+'.sf',3)
eps_nm1 = reconstitute(fn+'chkpt.'+nstr+'.sf',4)
t_nm1   = msgpack.load(open(fn+'chkpt.'+nstr+'.sf', 'rb'))[b'time']

rho0    =  rho_nm1*1 #Keep for buffer source terms
vx0     =   vx_nm1*1
vy0     =   vy_nm1*1
pres0   = pres_nm1*1
eps0    =  eps_nm1*1

n       = Nchkpts[1]
nstr    = str(np.char.zfill(str(n),4))
d_n     = msgpack.load(open(fn+'chkpt.'+nstr+'.sf', 'rb'))
rho_n   = reconstitute(fn+'chkpt.'+nstr+'.sf',0)
vx_n    = reconstitute(fn+'chkpt.'+nstr+'.sf',1)
vy_n    = reconstitute(fn+'chkpt.'+nstr+'.sf',2)
pres_n  = reconstitute(fn+'chkpt.'+nstr+'.sf',3)
eps_n   = reconstitute(fn+'chkpt.'+nstr+'.sf',4)
t_n     = d_n[b'time']
x1_n,y1_n   = d_n[b'masses'][0][b'x' ], d_n[b'masses'][0][b'y' ]
x2_n,y2_n   = d_n[b'masses'][1][b'x' ], d_n[b'masses'][1][b'y' ]
vx1_n,vy1_n = d_n[b'masses'][0][b'vx'], d_n[b'masses'][0][b'vy']
vx2_n,vy2_n = d_n[b'masses'][1][b'vx'], d_n[b'masses'][1][b'vy']

t = []
t.append(t_n)

n       = Nchkpts[2]
nstr    = str(np.char.zfill(str(n),4))
d_np1   = msgpack.load(open(fn+'chkpt.'+nstr+'.sf', 'rb'))
rho_np1 = reconstitute(fn+'chkpt.'+nstr+'.sf',0)
vx_np1  = reconstitute(fn+'chkpt.'+nstr+'.sf',1)
vy_np1  = reconstitute(fn+'chkpt.'+nstr+'.sf',2)
pres_np1= reconstitute(fn+'chkpt.'+nstr+'.sf',3)
eps_np1 = reconstitute(fn+'chkpt.'+nstr+'.sf',4)
t_np1   = d_np1[b'time']
x1_np1,y1_np1   = d_np1[b'masses'][0][b'x' ], d_np1[b'masses'][0][b'y' ]
x2_np1,y2_np1   = d_np1[b'masses'][1][b'x' ], d_np1[b'masses'][1][b'y' ]
vx1_np1,vy1_np1 = d_np1[b'masses'][0][b'vx'], d_np1[b'masses'][0][b'vy']
vx2_np1,vy2_np1 = d_np1[b'masses'][1][b'vx'], d_np1[b'masses'][1][b'vy']


dt_rho_n, dt_momx_n, dt_momy_n, dt_en_n = compute_dert( rho_nm1, rho_np1,\
							 vx_nm1,  vx_np1,\
							 vy_nm1,  vy_np1,\
							eps_nm1, eps_np1,\
							  t_nm1,   t_np1)

d_rhoflux_n, d_momxflux_n, d_momyflux_n, d_enflux_n = compute_all_fluxes_der( rho_n, vx_n, vy_n, pres_n, eps_n, xx, yy, x1_n, y1_n, x2_n, y2_n, m1, m2 )

momxsrc_grav, momysrc_grav, ensrc_grav = sources_gravity( rho_n, pres_n, vx_n, vy_n, xx, yy, x1_n, y1_n, x2_n, y2_n )

rhosrc_buff, momxsrc_buff, momysrc_buff, ensrc_buff = sources_buffer( rho_n, vx_n, vy_n, pres_n, eps_n, rho0, vx0, vy0, pres0, eps0, xx, yy )

ensrc_cool = sources_cooling( rho_n, eps_n )

rhosrc_sink1, momxsrc_sink1, momysrc_sink1, ensrc_sink1 = sources_sinks( rho_n, vx_n, vy_n, pres_n, eps_n, xx, yy, x1_n, y1_n, vx1_n, vy1_n )
rhosrc_sink2, momxsrc_sink2, momysrc_sink2, ensrc_sink2 = sources_sinks( rho_n, vx_n, vy_n, pres_n, eps_n, xx, yy, x2_n, y2_n, vx2_n, vy2_n )

rhores   ,momxres   ,momyres   ,enres    = [],[],[],[]
rhores_L2,momxres_L2,momyres_L2,enres_L2 = [],[],[],[]

rhores. append( dt_rho_n  + d_rhoflux_n  -  rhosrc_buff -  rhosrc_sink1 - rhosrc_sink2                                )
momxres.append( dt_momx_n + d_momxflux_n - momxsrc_buff - momxsrc_grav  - momxsrc_sink1 - momxsrc_sink2               )
momyres.append( dt_momy_n + d_momyflux_n - momysrc_buff - momysrc_grav  - momysrc_sink1 - momysrc_sink2               )
enres.  append( dt_en_n   + d_enflux_n   -   ensrc_buff -   ensrc_grav  - ensrc_cool    -   ensrc_sink1 - ensrc_sink2 )

l,r=0,N

rhores_L2. append( np.sqrt(np.average(( rhores[-1][l:r,l:r]*buffer_mask)**2)) )
momxres_L2.append( np.sqrt(np.average((momxres[-1][l:r,l:r]*buffer_mask)**2)) )
momyres_L2.append( np.sqrt(np.average((momyres[-1][l:r,l:r]*buffer_mask)**2)) )
enres_L2.  append( np.sqrt(np.average((  enres[-1][l:r,l:r]*buffer_mask)**2)) )

for i in range(3,len(Nchkpts)-1):
	print("Analyzing checkpoint",i,"of",len(Nchkpts)-2)

	rho_nm1 =  rho_n*1
	vx_nm1  =   vx_n*1
	vy_nm1  =   vy_n*1
	pres_nm1= pres_n*1
	eps_nm1 =  eps_n*1
	t_nm1   =    t_n*1

	rho_n   =  rho_np1*1
	vx_n    =   vx_np1*1
	vy_n    =   vy_np1*1
	pres_n  = pres_np1*1
	eps_n   =  eps_np1*1
	t_n     =    t_np1*1
	t.append(t_n)
	x1_n,y1_n = x1_np1*1,y1_np1*1
	x2_n,y2_n = x2_np1*1,y2_np1*1
	vx1_n,vy1_n = vx1_np1*1,vy1_np1*1
	vx2_n,vy2_n = vx2_np1*1,vy2_np1*1

	n       = Nchkpts[i]
	nstr    = str(np.char.zfill(str(n),4))
	d_np1   = msgpack.load(open(fn+'chkpt.'+nstr+'.sf','rb'))
	rho_np1 = reconstitute(fn+'chkpt.'+nstr+'.sf',0)
	vx_np1  = reconstitute(fn+'chkpt.'+nstr+'.sf',1)
	vy_np1  = reconstitute(fn+'chkpt.'+nstr+'.sf',2)
	pres_np1= reconstitute(fn+'chkpt.'+nstr+'.sf',3)
	eps_np1 = reconstitute(fn+'chkpt.'+nstr+'.sf',4)
	t_np1   = d_np1[b'time']
	x1_np1,y1_np1   = d_np1[b'masses'][0][b'x' ], d_np1[b'masses'][0][b'y' ]
	x2_np1,y2_np1   = d_np1[b'masses'][1][b'x' ], d_np1[b'masses'][1][b'y' ]
	vx1_np1,vy1_np1 = d_np1[b'masses'][0][b'vx'], d_np1[b'masses'][0][b'vy']
	vx2_np1,vy2_np1 = d_np1[b'masses'][1][b'vx'], d_np1[b'masses'][1][b'vy']

	dt_rho_n, dt_momx_n, dt_momy_n, dt_en_n = compute_dert( rho_nm1, rho_np1,\
								 vx_nm1,  vx_np1,\
								 vy_nm1,  vy_np1,\
								eps_nm1, eps_np1,\
								  t_nm1,   t_np1)

	d_rhoflux_n, d_momxflux_n, d_momyflux_n, d_enflux_n = compute_all_fluxes_der( rho_n, vx_n, vy_n, pres_n, eps_n, xx, yy, x1_n, y1_n, x2_n, y2_n, m1, m2 )

	momxsrc_grav, momysrc_grav, ensrc_grav = sources_gravity( rho_n, pres_n, vx_n, vy_n, xx, yy, x1_n, y1_n, x2_n, y2_n )

	rhosrc_buff, momxsrc_buff, momysrc_buff, ensrc_buff = sources_buffer( rho_n, vx_n, vy_n, pres_n, eps_n, rho0, vx0, vy0, pres0, eps0, xx, yy )

	ensrc_cool = sources_cooling( rho_n, eps_n )

	rhosrc_sink1, momxsrc_sink1, momysrc_sink1, ensrc_sink1 = sources_sinks( rho_n, vx_n, vy_n, pres_n, eps_n, xx, yy, x1_n, y1_n, vx1_n, vy1_n )
	rhosrc_sink2, momxsrc_sink2, momysrc_sink2, ensrc_sink2 = sources_sinks( rho_n, vx_n, vy_n, pres_n, eps_n, xx, yy, x2_n, y2_n, vx2_n, vy2_n )

	rhores, momxres, momyres, enres = [],[],[],[] #kill old lists, otherwise memory usage grows

	rhores. append( dt_rho_n  + d_rhoflux_n  -  rhosrc_buff -  rhosrc_sink1 - rhosrc_sink2                                )
	momxres.append( dt_momx_n + d_momxflux_n - momxsrc_buff - momxsrc_grav  - momxsrc_sink1 - momxsrc_sink2               )
	momyres.append( dt_momy_n + d_momyflux_n - momysrc_buff - momysrc_grav  - momysrc_sink1 - momysrc_sink2               )
	enres.  append( dt_en_n   + d_enflux_n   -   ensrc_buff -   ensrc_grav  - ensrc_cool    -   ensrc_sink1 - ensrc_sink2 )

	rhores_L2. append( np.sqrt(np.average(( rhores[-1][l:r,l:r]*buffer_mask)**2)) )
	momxres_L2.append( np.sqrt(np.average((momxres[-1][l:r,l:r]*buffer_mask)**2)) )
	momyres_L2.append( np.sqrt(np.average((momyres[-1][l:r,l:r]*buffer_mask)**2)) )
	enres_L2.  append( np.sqrt(np.average((  enres[-1][l:r,l:r]*buffer_mask)**2)) )

rhores_L2  = np.array( rhores_L2)
momxres_L2 = np.array(momxres_L2)
momyres_L2 = np.array(momyres_L2)
enres_L2   = np.array(  enres_L2)
t          = np.array(t)
