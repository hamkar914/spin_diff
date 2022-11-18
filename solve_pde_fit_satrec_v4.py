'''Test script written by Hampus Karlsson, August 2021.
 e-mail: hamka@chalmers.se/hkarlsson914@gmail.com
 Solves diffusion equation using Crank-Nicholson scheme,
 for analysis and simulation of ssNMR spin-diffusion
 saturation recovery data.

 Version 4 implements:
     - space dependent diffusion constant
     - space dependent equilibrium polarizations
     - improved handling of boundary conditions
     - possibility of least squares fit of experimental data
     - Improved integration of solution to calculate signal

 Literature:
 1. Schlagnitweit, J. et al. J.A.C.S, 137, pp. 12482-12485 (2015)
 2. Chapter 19. in Numerical Recipes in C, 2nd ed.
 3. http://www.claudiobellei.com/2016/11/01/implicit-parabolic/
 4. http://www.claudiobellei.com/2016/11/10/crank-nicolson/
 5. Novikov, E.G. et. al., J.Mag.Res. 135, pp. 522-528 (1998)
 6. https://www.youtube.com/watch?v=lgEOBBNtjiI&t=1377s
 7. Pinon, A.C. Spin Diffusion in Dynamic Nuclear Polarization
    Nuclear Magnetic Resonance. EPFL Phd thesis no. 8519 (2018)
 '''


import numpy as np
from scipy.optimize import least_squares
from scipy.integrate import trapz
import matplotlib.pyplot as plt


def solve_pde(Dr,Dp,p0r,p0p,t1r,t1p,dt,N,t,sp,Cpr,Cpp,dx):

    '''Solves the diffusion equation (Ref.1) using
    the Crank-Nicholson method (Ref.2-6). Certain
    terms are based on (Ref. 7).

    Arguments:
    R = radius of particle (m)
    Dr = diffusion constant radical soluton (m2/s)
    Dp = diffusion constant particle (m2/s)
    p0r = equilibrium polarization in radical solution
    p0p = equilibrium polarization in particle
    t1r = T1 relaxation time in radical solution (s)
    t1p = T1 relaxation time in particle (s)
    dt = time step
    N = number of grid squares
    t = time, how long to evolve solution (s)
    sp = start particle, specifies at what grid square
    Dp and t1p starts to apply
    Cpr = proton concentration radical
    Cpp = proton concentration particle'''

    # Create tridiagonal Matrix A,B (see ref.3-4)
    A = np.zeros((N,N))
    B = np.zeros((N,N))

    # Define column vector C to hold constant term P0(x)/T1(x)
    C = np.zeros((N))

    # Define array to hold solution
    soln = np.zeros((int(t/dt)+1,N))

    # dx = grid square size in spatial dimension
    # depends on particle radius and N
    r = dt/(2*dx**2)

    # Fill tri-diagonal matrices A and B and column vector C
    # use different T1 for radical and particle
    for m in range(N):

        # fill column vector C
        if m < sp:
            C[m] = (Cpr*dt*p0r)/t1r
        elif sp <= m:
            C[m] = (Cpp*dt*p0p)/t1p

        for n in range(N):

            if m == n and m == 0:
                # Upper left corner of matrix
                # Expressions here take care of
                # BC: dP(x,t)/dx = 0, see Ref.6
                A[m,n] = Cpr+(2/3)*r*Dr*Cpr+(dt*Cpr/t1r)
                A[m,n+1] = -(2/3)*r*Dr*Cpr

                B[m,n] = Cpr-(2/3)*r*Dr*Cpr
                B[m,n+1] = (2/3)*r*Dr*Cpr

            elif m == n and 0 < m < sp:
                # From corner until border of particle
                A[m,n] = Cpr+2*r*Dr*Cpr+(dt*Cpr/t1r)
                A[m,n-1] = -r*Dr*Cpr
                A[m,n+1] = -r*Dr*Cpr

                B[m,n] = Cpr-2*r*Dr*Cpr
                B[m,n-1] = r*Dr*Cpr
                B[m,n+1] = r*Dr*Cpr

            elif m == n and m == sp:
                # At start border radical/particle
                A[m,n] = Cpp+r*Dp*Cpp+r*Dr*Cpr+(dt*Cpp/t1p)
                A[m,n-1] = -r*Dr*Cpr
                A[m,n+1] = -r*Dp*Cpp

                B[m,n] = Cpp-r*Dp*Cpp-r*Dr*Cpr
                B[m,n-1] = r*Dr*Cpr
                B[m,n+1] = r*Dp*Cpp

            elif m == n and sp < m <N-1:
                # Inside particle
                A[m,n] = Cpp+2*r*Dp*Cpp+(dt*Cpp/t1p)
                A[m,n-1] = -r*Dp*Cpp
                A[m,n+1] = -r*Dp*Cpp

                B[m,n] = Cpp-2*r*Dp*Cpp
                B[m,n-1] = r*Dp*Cpp
                B[m,n+1] = r*Dp*Cpp

            elif m == n and m == N - 1:
                # Lower right corner
                # Also here BC applies
                A[m,n] = Cpp+(2/3)*r*Dp*Cpp+(dt*Cpp/t1p)
                A[m,n-1] = -(2/3)*r*Dp*Cpp

                B[m,n] = Cpp-(2/3)*r*Dp*Cpp
                B[m,n-1] = (2/3)*r*Dp*Cpp

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

    # Define array to hold solution multiplied
    # by jacobian determinant for spherical symmetry.
    sphere_soln = np.zeros((int(t/dt)+1,N))

    # Jacobian determinant for sphere (Ref.7 p.43)
    Jac = np.asarray([4*np.pi*x**2 for x in np.arange(N*dx,0,-dx)])

    # 1D array with space dependent concentration
    cofx = np.concatenate((np.full((sp),Cpr),np.full((N-sp),Cpp)))

    # Multiply by Jac, see eq.2.15 (Ref.7)
    for k in range(int(t/dt)+1):
        #sphere_soln[k,:] = np.multiply(np.multiply(cofx,soln[k,:]),Jac)
        sphere_soln[k,:] = np.multiply(soln[k,:],Jac)

    return (soln,dx)


def t1_func(t,I0,T1):
    '''helpful function describing
    return from saturation'''
    return I0*(1.0-np.exp(-t/T1))


def t1_func_stretch(t,I0,T1,B):
    '''Helpful function describing
    return from saturation stretched
    /compressed exponential'''
    return I0*(1.0-np.exp(-(t/T1)**B))


def t1_func_biexp(t,I0,T1a,T1c,fc):
    '''helpful function describing biexponential
        return from saturation'''
    return I0*(fc*(1.0-np.exp(-t/T1c))+(1.0-fc)*(1.0-np.exp(-t/T1a)))


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
        i0,t1 = popt
        t, = fixed_args
        y_calc = t1_func(t,i0,t1)
        sq_res = (y_exp-y_calc)**2
        return sq_res

    elif obj_func == "t1b":
        i0,t1,beta = popt
        t, = fixed_args
        y_calc = t1_func_stretch(t, i0, t1, beta)
        sq_res = (y_exp - y_calc) ** 2
        return sq_res

    elif obj_func == "t1bi":
        i0,t1a,t1c,fc = popt
        t, = fixed_args
        y_calc = t1_func_biexp(t, i0, t1a, t1c, fc)
        sq_res = (y_exp - y_calc) ** 2
        return sq_res

    elif obj_func == "pde":

        # Fit particle radius
        r = popt

        # unpack all fixed arguments, "ind"
        # are indices for experimental time points
        Dr,Dp,pr,pp,t1r,t1p,dt,N,time,sp,ind = fixed_args
        # Dr, Dp, pr, pp, t1r, t1p, dt, N, t, sp, calc_indexes(vd_list,t,dt)
        # R,Dr,Dp,p0r,p0p,t1r,t1p,dt,N,t,sp

        sim = solve_pde(r[0],Dr,Dp,pr,pp,t1r,t1p,dt,N,time,sp)

        y_calc = []

        # loop through calculated curve and pick out points
        # corresponding to experimental vd list points,
        for i in ind:
            y_calc.append(sim[1][i])

        y_calc = np.asarray(y_calc)
        sq_res = np.sum((y_exp-(y_calc/y_calc.max()))**2)

        print(str("{:.3g}".format(r[0]))+"\t"+str(round(sq_res,3)))

        return sq_res


# -------------------------
# Parameters:
# -------------------------

R = 1000E-9          # Particle radius (m)
Dr = 1E-15           # diffusion constant radical (m2/s)
Dp = 1E-15           # diffusion constant particle (m2/s)
pr = 100             # Equilibrium polarization in radical solution
pp = 1.0             # Equilibrium polarization particle
t1r = 0.7            # Short T1 (s) inside radical solution
t1p = 5.0            # Long T1 (s) inside cellulose particle
dt = 0.01            # Time step (s)
N = 100              # Number of grid squares
t = 200.0            # Total simulation time (s)
sp = 50              # array index start of particle
CHr = 11.0           # [1H] in radical solution (M)
CHp = 136.0          # [1H] in particle (M)
dx = R/(N-sp)

# -------------------------
# Simulate/plot:
# -------------------------


# def solve_pde(Dr,Dp,p0r,p0p,t1r,t1p,dt,N,t,sp,Cpr,Cpp,dx):
sim = solve_pde(Dr,Dp,pr,pp,t1r,t1p,dt,N,t,sp,CHr,CHp,dx)

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

tick_labels = [str("{:.2g}".format((x)*sim[1])) for x in tick_factors]
plt.xticks(tick_pos,tick_labels,rotation=45)
plt.xlabel("Dimension (m)")
plt.axvline(sp,color="k",linestyle="--")
plt.colorbar()
plt.tight_layout()
plt.show()


# 2. Plot spatial profiles
# ---------------------------
decs = calc_indexes(np.array([0.5,1.0,2.5,5.0,10.0,15.0,20.0,30.0,199.0]),t,dt)
fig = plt.figure()
plt.plot(sim[0][decs[0],],label="0.5 s")
plt.plot(sim[0][decs[1],],label="1 s")
plt.plot(sim[0][decs[2],],label="2.5 s")
plt.plot(sim[0][decs[3],],label="5 s")
plt.plot(sim[0][decs[4],],label="10 s")
plt.plot(sim[0][decs[5],],label="15 s")
plt.plot(sim[0][decs[6],],label="20 s")
plt.plot(sim[0][decs[7],],label="30.0 s")
plt.plot(sim[0][decs[8],],label="199.0 s")

plt.axvspan(0, sp, facecolor='royalblue', alpha=0.2)
plt.axvspan(sp, N, facecolor='lightcoral', alpha=0.2)
plt.axvline(sp,color="gray",linestyle="--")
plt.xticks(tick_pos,tick_labels,rotation=45)
plt.xlim(0.0,N)
plt.xlabel("Dimension (m)")
plt.ylabel("Polarization (a.u.)")
plt.tight_layout()
plt.legend()
plt.show()
fig.savefig("relaxation_profiles.eps",format="eps",dpi=300)

