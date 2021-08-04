'''Test script written by Hampus Karlsson, July 2021.
 e-mail: hamka@chalmers.se/hkarlsson914@gmail.com
 Solves diffusion equation using Crank-Nicholson scheme,
 for analysis and simulation of ssNMR spin-diffusion
 saturation recovery data.

 Literature:
 1. Schlagnitweit, J. et al. J.A.C.S, 137, pp. 12482-12485 (2015)
 2. Chapter 19. in Numerical Recipes in C, 2nd ed.
 3. http://www.claudiobellei.com/2016/11/01/implicit-parabolic/
 4. http://www.claudiobellei.com/2016/11/10/crank-nicolson/
 5. Novikov, E.G. et. al., J.Mag.Res. 135, pp. 522-528 (1998)
 '''


import numpy as np
import matplotlib.pyplot as plt


def solve_pde(d,D,t1l,t1s,dt,N,t,s_p,e_p):

    '''Solves the diffusion equation (Ref.1) using the Crank-Nicholson
    method (Ref.2-5). Returns (t/dt)*N array where each row is a time
    point (1D array).'''

    # Define array to hold solution
    soln = np.zeros((int(t/dt)+1,int(N)))

    # Matrix A,B (see ref.3-4)
    A = np.zeros((N,N))
    B = np.zeros((N,N))

    # grid square size in spatial dimension
    # depends on particle diameter and N
    # N is assumed to always be 100
    dx = d/(e_p-s_p)
    r = D/(2*dx**2)
    # eql. polarization "P0"
    P0 = 1.0

    # s_p = start of particle = the position in 1D array where long t1 star to apply
    # e_p = end of particle = the position in 1D array where long t1 ends

    # column vector C that will contain constant term P0/T1
    C = np.zeros((N))

    # Fill tri-diagonal matrices A and B and column vector C
    # use different T1 for AMUPol and particle
    for m in range(N):

        # fill column vector C
        if m < s_p:
            C[m] = (dt*P0)/t1s
        elif s_p <= m <= e_p:
            C[m] = (dt*P0)/t1l
        elif e_p < m:
            C[m] = (dt*P0)/t1s

        for n in range(N):

            if m == n and m == 0:
                # upper left corner of matrix
                A[m,n] = 2*dt*r+(dt/t1s)+1
                A[m,n+1] = -dt*r

                B[m,n] = -2*dt*r+1
                B[m,n+1] = dt*r

            elif m == n and 0 < m < s_p:
                A[m,n] = 2*dt*r+(dt/t1s)+1
                A[m,n-1] = -dt*r
                A[m,n+1] = -dt*r

                B[m, n] = -2*dt*r+1
                B[m, n - 1] = dt*r
                B[m, n + 1] = dt*r

            elif m == n and s_p <= m <= e_p:
                # inside particle
                A[m,n] = 2*dt*r+(dt/t1l)+1
                A[m,n-1] = -dt*r
                A[m,n+1] = -dt*r

                B[m,n] = -2*dt*r+1
                B[m,n-1] = dt*r
                B[m,n+1] = dt*r

            elif m == n and e_p < m < N - 1:
                A[m,n] = 2*dt*r+(dt/t1s)+1
                A[m, n - 1] = -dt*r
                A[m, n + 1] = -dt*r

                B[m,n] = -2*dt*r+1
                B[m,n-1] = dt*r
                B[m,n+1] = dt*r

            elif m == n and m == N - 1:
                A[m,n] = 2*dt*r+(dt/t1s)+1
                A[m,n-1] = -dt*r

                B[m,n] = -2*dt*r+1
                B[m,n-1] = dt*r


    # Now solve the system for each dt
    for i in range(int(t/dt)):

        # the known time point
        u0 = np.copy(soln[i,:])

        # multiply known time point with B
        Bu0 = np.dot(B,u0.T)

        # Add column vector C to rhs
        Bu0pC = Bu0+C

        # invert A of lhs
        Ainv = np.linalg.inv(A)

        # multiply Bu0mC with A invert to get next point in time
        u1 = np.dot(Ainv, Bu0pC)

        # enforce boundary condition, i.e. set outermost grid squares
        # equal to second outermost square = no gradient, du(x,t)/dt=0
        u1[0] = u1[1]
        u1[-1] = u1[-2]

        # save the next point in time in solution array at row +1
        soln[i+1] = u1

    # Calculate peak buildup i.e.
    # sum all grid squares belonging to
    # particle limited by s_p and e_p
    # and normalize
    pint = np.zeros((int(t/dt)))

    for k in range(int(t/dt)):
        pint[k]=np.sum(soln[k,s_p:e_p])/(e_p-s_p)

    return (soln,pint,dx)


def t1_func(t,I0,T1):
    # helpful function describing
    # return from saturation
    return I0*(1.0-np.exp(-t/T1))


# Parameters for simulation
# -------------------------
d = 1E-6         # particle diameter
D = 1E-15        # diffusion constant m2/s (Ref.1)
t1l = 3.0        # T1 long (s) inside cellulose particle
t1s = 1.0        # T1 short (s) inside AMUPol where PRE applies
dt = 0.01        # time step (s)
N = 100          # number of grid squares space dimension (100)
t = 10.0         # Total simulation time (s)
sp = 40          # grid square start of particle
ep = 60          # grid square end of particle


# Simulate and plot some things
# -------------------------
# solve_pde(d,D,t1l,t1s,dt,N,t):
evo = solve_pde(d,D,t1l,t1s,dt,N,t,sp,ep)

xt = np.arange(0.0,evo[2]*N,evo[2])
yt = np.arange(0.0,t,dt)

plt.imshow(evo[0],aspect="auto")
plt.yticks(np.arange(0,int(t/dt),100),[str(float(x)) for x in range(int(t))])
plt.ylabel("Time (s)")

plt.xticks(np.arange(0,N,N/5),[str(float(x)) for x in evo[2]*np.arange(0,N,N/5)])
plt.xlabel("Dimension (m)")

plt.axvline(sp,color="gray",linestyle="--")
plt.axvline(ep,color="gray",linestyle="--")
plt.colorbar()
plt.show()


# Plot build up diff eq. vs. t1 recovery
plt.plot(np.arange(0.0,t,dt),solve_pde(d,D,t1l,t1s,dt,N,t,sp,ep)[1],"b-",label="Diffusion eq.")
plt.plot(np.arange(0.0,t,dt),t1_func(np.arange(0.0,t,dt),1.0,t1l),"k--",label="t1 eq.")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Normalized peak integral (a.u.)")
plt.show()


# Plot build up diff sizes
plt.plot(np.arange(0.0,t,dt),solve_pde(0.5E-6,D,t1l,t1s,dt,N,t,sp,ep)[1],"r-",label="0.5 um")
plt.plot(np.arange(0.0,t,dt),solve_pde(1E-6,D,t1l,t1s,dt,N,t,sp,ep)[1],"g-",label="1 um")
plt.plot(np.arange(0.0,t,dt),solve_pde(3E-6,D,t1l,t1s,dt,N,t,sp,ep)[1],"b-",label="3 um")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Normalized peak integral (a.u.)")
plt.show()
