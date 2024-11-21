#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 13:46:05 2024

@author: elirusso
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2
from scipy.integrate import odeint
from scipy.integrate import solve_ivp

#-------------------------------------PART A-----------------------------------

# Define parameters
tspan = np.arange(0, 4.5, 0.5)
nu = 0.001
Lx, Ly = 20, 20
nx, ny = 64, 64
N = nx * ny

# Define spatial domain and initial conditions
x2 = np.linspace(-Lx/2, Lx/2, nx + 1)
x = x2[:nx]
y2 = np.linspace(-Ly/2, Ly/2, ny + 1)
y = y2[:ny]
X, Y = np.meshgrid(x, y)
w = np.exp(-X**2 - Y**2 / 20)
#w2 = w.reshape(N)
wt2 = w.flatten()


# Define spectral k values
kx = (2 * np.pi / Lx) * np.concatenate((np.arange(0, nx/2), np.arange(-nx/2, 0)))
kx[0] = 1e-6
ky = (2 * np.pi / Ly) * np.concatenate((np.arange(0, ny/2), np.arange(-ny/2, 0)))
ky[0] = 1e-6
KX, KY = np.meshgrid(kx, ky)
K = KX**2 + KY**2


"""
---------------------------------
"""
import numpy as np
from scipy.sparse import spdiags, diags
import matplotlib.pyplot as plt

# Define parameters
m = 64
n = m * m # N value in x and y directions
x_min, x_max = -10, 10
dx = (x_max - x_min) / m

# total size of matrix
e0 = np.zeros(n) # vector of zeros
e1 = np.ones(n) # vector of ones
e2 = np.copy(e1) # copy the one vector
e4 = np.copy(e0) # copy the zero vector

for j in range(1, m+1):
    e2[m*j-1] = 0 # overwrite every m^th value with zero
    e4[m*j-1] = 1 # overwirte every m^th value with one

# Shift to correct positions
e3 = np.zeros_like(e2)
e3[1:n] = e2[0:n-1]
e3[0] = e2[n-1]
e5 = np.zeros_like(e4)
e5[1:n] = e4[0:n-1]
e5[0] = e4[n-1]

# Place diagonal elements
diagonals = [e1.flatten(), e1.flatten(), e5.flatten(),e2.flatten(),-4*e1.flatten(), e3.flatten(),e4.flatten(), e1.flatten(), e1.flatten()]
offsets = [-(n-m), -m, -m+1, -1, 0, 1, m-1, m, (n-m)]
A = spdiags(diagonals, offsets, n, n).toarray()
A = A/(dx**2)

# Create matrix B
e = np.ones(n)

data_B = [e, -1*e, e, -1*e]
offsets_B = [-(n-m), -m, m, (n - m)]
B = spdiags(data_B, offsets_B, n, n).toarray()
B = B/(2*dx)

# Create matrix C
e1 = np.zeros(n)
e2 = np.ones(n)
e3 = np.ones(n)
e4 = np.zeros(n)


for i in range(n):
    if (i + 1) % m == 1:
        e1[i] = 1
        e3[i] = 0
    if (i + 1) % m == 0:
        e2[i] = 0
        e4[i] = 1

e2 = -1*e2
e4 = -1*e4

C = spdiags([e1,e2, e3,e4],[-m+1, -1,1,m-1], n,n).toarray()
C = C/(2*dx)

#print(A)
#print(B)
#print(C)

"""
---------------------------------
"""


# Step 5: Define the PDE
def pde(t, wt2, nx, ny, N, K, A, B, C, nu):
    # Reshape w to 2D for FFT operations
    w = wt2.reshape((nx, ny))
    
    # Compute Fourier Transform of w
    wtfft = fft2(w)
    
    psi = np.real(ifft2(- wtfft / K)).flatten()
    
    # Compute RHS
    rhs = (nu * (A @ wt2) + (B @ wt2) * (C @ psi) - (B @ psi) * (C @ wt2))
    
    return rhs

# Step 6: Solve the PDE using solve_ivp
solution = solve_ivp(
    pde,                        
    [0, 4],                          
    wt2,                             
    t_eval=tspan,                    
    args=(nx, ny, N, K, A, B, C, nu), 
    method='RK45'                   
)

# Extract solution
A1 = solution.y

# Output for verification
print("Shape of A1:", A1.shape, "\n")  # Should be (4096, 9)
print("A1 matrix:", A1, "\n")

# Adjust layout to accommodate all subplots
num_plots = len(tspan)
cols = 3  # Number of columns
rows = (num_plots + cols - 1) // cols  # Calculate the required number of rows

# Reshape results and plot
for i, t in enumerate(solution.t):
    omega_t = solution.y[:, i].reshape((nx, ny))  # Reshape the solution for current time step
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, omega_t, levels=50, cmap='viridis')  # Contour plot
    plt.colorbar(label='Vorticity')
    plt.title(f'Vorticity Contour at t={t:.1f}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    
    
#----------------------------------------------------------------------------

import numpy as np
from scipy.sparse import spdiags
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time


nu = 0.01
Lx = 20
Ly = 20
nx = 64
ny = 64
N = nx * ny
t = np.arange(0, 4.5, 0.5)  # Corrected function name


# Define spatial domain and initial conditions
x2 = np.linspace(-Lx/2, Lx/2, nx + 1)
x = x2[:nx]
y2 = np.linspace(-Ly/2, Ly/2, ny + 1)
y = y2[:ny]
X, Y = np.meshgrid(x, y)


w = np.exp(-X**2 - Y**2 / 20)
wt2 = w.reshape((N))

def abs_rhs(t, wt2, nx, ny, A, B, C, nu):
    psi = np.linalg.solve(A,wt2)
    rhs = nu * np.dot(A, wt2) + (np.dot(B, wt2) * np.dot(C, psi)) - (np.dot(B, psi) * np.dot(C, wt2))
    return rhs

absol = solve_ivp(abs_rhs, [t[0], t[-1]], wt2, t_eval=t, args = (nx, ny, A, B, C, nu), method = 'RK45')

A2 = absol.y

print(A2)


#----------------------------------------------------------------------------


import numpy as np
from scipy.integrate import solve_ivp
import time
from scipy.linalg import lu, solve_triangular

nu = 0.01
Lx = 20
Ly = 20
nx = 64
ny = 64
N = nx * ny
t = np.arange(0, 4.5, 0.5)  # Corrected function name


# Define spatial domain and initial conditions
x2 = np.linspace(-Lx/2, Lx/2, nx + 1)
x = x2[:nx]
y2 = np.linspace(-Ly/2, Ly/2, ny + 1)
y = y2[:ny]
X, Y = np.meshgrid(x, y)

w = np.exp(-X**2 - Y**2 / 20)
wt2 = w.reshape((N))

P, L, U = lu(A)

def lu_rhs(t, wt2, P, L, U, B, C, nu, A):
    Pb = np.dot(P, wt2)
    y = solve_triangular(L, Pb, lower=True)
    psi = solve_triangular(U, y)
    rhs = nu * np.dot(A, wt2) + (np.dot(B, wt2) * np.dot(C, psi)) - (np.dot(B, psi) * np.dot(C, wt2))
    return rhs

lusol = solve_ivp(lu_rhs, [t[0], t[-1]], wt2, t_eval=t, args = (P, L, U, A, B, C, nu), method = 'RK45')

A3 = lusol.y
print(A3)


    
    
    
    
    