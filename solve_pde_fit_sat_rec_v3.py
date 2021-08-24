'''Test script written by Hampus Karlsson, August 2021.
 e-mail: hamka@chalmers.se/hkarlsson914@gmail.com
 Solves diffusion equation using Crank-Nicholson scheme,
 for analysis and simulation of ssNMR spin-diffusion
 saturation recovery data.

 Version 3 implements:
     - space dependent diffusion constant
     - space dependent equilibrium polarizations
     - improved handling of boundary conditions
     - possibility of least squares fit of experimental data

 Literature:
 1. Schlagnitweit, J. et al. J.A.C.S, 137, pp. 12482-12485 (2015)
 2. Chapter 19. in Numerical Recipes in C, 2nd ed.
 3. http://www.claudiobellei.com/2016/11/01/implicit-parabolic/
 4. http://www.claudiobellei.com/2016/11/10/crank-nicolson/
 5. Novikov, E.G. et. al., J.Mag.Res. 135, pp. 522-528 (1998)
 6. https://www.youtube.com/watch?v=lgEOBBNtjiI&t=1377s
 '''


import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt


def solve_pde(R,Dr,Dp,p0r,p0p,t1r,t1p,dt,N,t,sp):

    '''Solves the diffusion equation (Ref.1) using
    the Crank-Nicholson method (Ref.2-6).

    Arguments:
    R = radius of particle (m)
    Dr = spin diffusion constant in radical solution (m^2/s)
    Dp = spin diffusion constant in particle (m^2/s)
    p0r = equilibrium polarization in radical solution
    p0p = equilibrium polarization in particle
    t1r = T1 relaxation time in radical solution (s)
    t1p = T1 relaxation time in particle (s)
    dt = time step
    N = number of grid squares
    t = time, how long to evolve solution (s)
    sp = start particle, specifies at what grid square
    Dp and t1p starts to apply.'''

    # Create tridiagonal Matrix A,B (see ref.3-4)
    A = np.zeros((N,N))
    B = np.zeros((N,N))

    # Define column vector C to hold constant term P0(x)/T1(x)
    C = np.zeros((N))

    # Define array to hold solution
    soln = np.zeros((int(t/dt)+1,N))

    # dx = grid square size in spatial dimension
    # depends on particle radius and N
    dx = R/(N-(sp+1))
    r = dt/(2*dx**2)

    # Fill tri-diagonal matrices A and B and column vector C
    # use different T1 for radical and particle
    for m in range(N):

        # fill column vector C
        if m < sp:
            C[m] = (dt*p0r)/t1r
        elif sp <= m:
            C[m] = (dt*p0p)/t1p

        for n in range(N):

            if m == n and m == 0:
                # Upper left corner of matrix
                # Expressions here take care of
                # BC: dP(x,t)/dx = 0, see Ref.6
                A[m,n] = 1+(2/3)*r*Dr+(dt/t1r)
                A[m,n+1] = -(2/3)*r*Dr

                B[m,n] = 1-(2/3)*r*Dr
                B[m,n+1] = (2/3)*r*Dr

            elif m == n and 0 < m < sp:
                # From corner until border of particle
                A[m,n] = 1+2*r*Dr+(dt/t1r)
                A[m,n-1] = -r*Dr
                A[m,n+1] = -r*Dr

                B[m, n] = 1-2*r*Dr
                B[m, n - 1] = r*Dr
                B[m, n + 1] = r*Dr

            elif m == n and m == sp:
                # At start border radical/particle
                A[m,n] = 1+r*Dp+r*Dr+(dt/t1p)
                A[m,n-1] = -r*Dr
                A[m,n+1] = -r*Dp

                B[m,n] = 1-r*Dp-r*Dr
                B[m,n-1] = r*Dr
                B[m,n+1] = r*Dp

            elif m == n and sp < m <N-1:
                # Inside particle
                A[m,n] = 1+2*r*Dp+(dt/t1p)
                A[m,n-1] = -r*Dp
                A[m,n+1] = -r*Dp

                B[m,n] = 1-2*r*Dp
                B[m,n-1] = r*Dp
                B[m,n+1] = r*Dp

            elif m == n and m == N - 1:
                # Lower right corner
                # Also here BC applies
                A[m,n] = 1+(2/3)*r*Dp+(dt/t1p)
                A[m,n-1] = -(2/3)*r*Dp

                B[m,n] = 1-(2/3)*r*Dp
                B[m,n-1] = (2/3)*r*Dp


    # Invert A
    Ainv = np.linalg.inv(A)

    # Now solve the system for each dt
    for i in range(int(t/dt)):

        # The known time point
        u0 = np.copy(soln[i,:])

        # Multiply known time point with B
        Bu0 = np.dot(B,u0.T)

        # Add column vector C to rhs
        Bu0pC = Bu0+C

        # Multiply Bu0pC with A inverse to get next point in time
        u1 = np.dot(Ainv, Bu0pC)

        # Save the next point in time in solution array at row +1
        soln[i+1] = u1

    # Calculate peak buildup i.e.
    # sum all grid squares belonging to
    # particle and normalize
    pint = np.zeros((int(t/dt)+1))

    for k in range(int(t/dt)+1):
        pint[k]=np.sum(soln[k,sp:])/(p0p*(N-sp))

    return (soln,pint,dx)


def t1_func(t,I0,T1):
    '''helpful function describing
    return from saturation'''
    return I0*(1.0-np.exp(-t/T1))


def t1_func_stretch(t,I0,T1,B):
    '''Helpful function describing
    return from saturation stretched
    /compressed exponential'''
    return I0*(1.0-np.exp(-(t/T1)**B))


def calc_indexes(vdlist,t,dt):

    '''Helper function that calculates indices
    for time points in a vdlist'''

    indexes = np.zeros(vdlist.shape,dtype=int)
    t_points = np.arange(0.0, t + dt, dt)

    for h in range(vdlist.shape[0]):
        vdpnt = vdlist[h]
        for i in range(t_points.shape[0]):
            tpnt = t_points[i]
            if vdpnt == tpnt:
                indexes[h] = i

    return indexes


def residual(popt,y_exp,fixed_args,obj_func):

    '''For least square fits of experimental data to
    different functions.'''

    if obj_func == "t1":
        t1,i0 = popt
        t, = fixed_args
        y_calc = t1_func(t,i0,t1)
        sq_res = (y_exp-y_calc)**2
        return sq_res

    elif obj_func == "t1b":
        t1,beta,i0 = popt
        t, = fixed_args
        y_calc = t1_func_stretch(t, i0, t1,beta)
        sq_res = (y_exp - y_calc) ** 2
        return sq_res

    elif obj_func == "pde":

        # Fit particle radius
        r = popt

        # unpack all fixed arguments, "ind"
        # are indices for experimental time points
        Dr,Dp,pr,pp,t1r,t1p,dt,N,time,sp,ind = fixed_args
        sim = solve_pde(r[0],Dr,Dp,pr,pp,t1r,t1p,dt,N,time,sp)

        y_calc = []

        # loop through calculated curve and pick out points
        # corresponding to experimental vd list points,
        for i in ind:
            y_calc.append(sim[1][i])

        y_calc = np.asarray(y_calc)
        sq_res = np.sum((y_exp-(y_calc/y_calc.max()))**2)

        print(str("{:.3g}".format(r[0]))+"\t"+str(round(sq_res,3)))
        #plt.plot(sim[1]/sim[1].max(),"k-")
        #plt.plot(ind,y_exp,"bo")
        #plt.show()

        return sq_res


# -------------------------
# Parameters:
# -------------------------

R = 200E-9          # Particle radius (m)
Dr = 0.5E-15        # Diffusion constant radical solution (ca. 5E-16 m^2/s)
Dp = 0.8E-15        # Diffusion constant particle ca. 1E-15 m^2/s (Ref.1)
pr = 100.0          # Equilibrium polarization radical solution
pp = 1.0            # Equilibrium polarization particle
t1r = 0.1           # Short T1 (s) inside radical solution
t1p = 2.5           # Long T1 (s) inside cellulose particle
dt = 0.01           # Time step (s)
N = 300             # Number of grid squares
t = 10.0            # Total simulation time (s)
sp = 150            # array index start of particle


'''
# -------------------------
# Simulate/plot:
# -------------------------

# solve_pde(d,Dr,Dp,p0r,p0p,t1r,t1p,dt,N,t,sp):
sim = solve_pde(R,Dr,Dp,pr,pp,t1r,t1p,dt,N,t,sp)

# array with time points
time_points = np.arange(0.0,t+dt,dt)


# 1. Plot solution as 2D image
# ----------------------------
plt.imshow(sim[0],aspect="auto")
plt.yticks(np.arange(0,int(t/dt),100),[str(float(x)) for x in range(int(t))])
plt.ylabel("Time (s)")

# Calculate units x-axis
tick_pos = [x-1 for x in range(0,N+1,int(N/10)) if x!=0]
tick_pos.insert(0,0)

tick_factors = [x for x in range(0,N+1,int(N/10)) if x!=0]
tick_factors.insert(0,0)

tick_labels = [str("{:.2g}".format((x)*sim[2])) for x in tick_factors]
plt.xticks(tick_pos,tick_labels,rotation=45)
plt.xlabel("Dimension (m)")
plt.axvline(sp-1,color="k",linestyle="--")
plt.colorbar()
plt.tight_layout()
plt.show()

# 2. Plot and compare with exponentials
# -------------------------------------
plt.plot(time_points,sim[1],"b-")
plt.plot(time_points,t1_func(time_points,1.0,t1p),"k--")
plt.xlabel("Time (s)")
plt.ylabel("Normalized signal (a.u.)")
plt.show()

# 3. Plot spatial profiles
# ---------------------------
decs = calc_indexes(np.array([0.5,1.0,2.5,5.0,7.5,10.0]),t,dt)

plt.plot(sim[0][decs[0],],label="0.5 s")
plt.plot(sim[0][decs[1],],label="1.0 s")
plt.plot(sim[0][decs[2],],label="2.5 s")
plt.plot(sim[0][decs[3],],label="5.0 s")
plt.plot(sim[0][decs[4],],label="7.5 s")
plt.plot(sim[0][decs[5],],label="10.0 s")
plt.axvspan(0, sp, facecolor='royalblue', alpha=0.2)
plt.axvspan(sp, N, facecolor='lightcoral', alpha=0.2)
plt.axvline(sp-1,color="gray",linestyle="--")
plt.xticks(tick_pos,tick_labels,rotation=45)
plt.xlim(0.0,300.0)
plt.xlabel("Dimension (m)")
plt.ylabel("Polarization (a.u.)")
plt.tight_layout()
plt.legend()
plt.show()
'''

# -------------------------
# Least square fits
# -------------------------

# get experimental data
# -----------------------
inpf = "integrals.txt"
data = np.genfromtxt(inpf,skip_header=3)
vd_list = data[:,0]

# Take data and normalize
on_integrals = data[:,1]/data[:,1].max()
on_err = data[:,2]/data[:,2].max()
off_integrals = data[:,3]/data[:,3].max()
off_err = data[:,4]/data[:,4].max()

t_pnts = np.arange(vd_list.min(),vd_list.max()+dt,0.010)

# 1. Fit a particle diameter
p0 = [R]
fix_params = [Dr, Dp, pr, pp, t1r, t1p, dt, N, t, sp, calc_indexes(vd_list,t,dt)]
plsq = least_squares(residual, p0, args=(on_integrals[0:],fix_params,"pde"),bounds=([0.0],[np.inf]),method="trf")

# solve_pde(R,Dr,Dp,p0r,p0p,t1r,t1p,dt,N,t,sp):
fitted_soln3 = solve_pde(plsq.x[0],Dr,Dp,pr,pp,t1r,t1p,dt,N,t,sp)
plt.plot(t_pnts[:-1],fitted_soln3[1][:-1]/fitted_soln3[1][:-1].max(),"k-")
plt.plot(vd_list,on_integrals,"ro",mec="k",markersize=7)
plt.xlabel("Time (s)")
plt.ylabel("Normalized peak integral (a.u.)")
plt.title("Radius = "+str("{:.2e}".format(plsq.x[0]))+" m")
plt.show()
