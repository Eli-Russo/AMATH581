#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 15:00:58 2024

@author: elirusso
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2
from scipy.integrate import odeint
from scipy.integrate import solve_ivp

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
w2 = w.reshape(N)


# Define spectral k values
kx = (2 * np.pi / Lx) * np.concatenate((np.arange(0, nx/2), np.arange(-nx/2, 0)))
kx[0] = 1e-6
ky = (2 * np.pi / Ly) * np.concatenate((np.arange(0, ny/2), np.arange(-ny/2, 0)))
ky[0] = 1e-6
KX, KY = np.meshgrid(kx, ky)
K = KX**2 + KY**2

"""
#------------------------------------------------------------------------------
"""

import numpy as np
from scipy.sparse import spdiags, diags
import matplotlib.pyplot as plt

# Parameters
L = 20            # Spatial domain in both x and y directions (assumed square)
m = 64             # Grid points in x and y (m x m grid)
dx = L / m        # Grid spacing (same for x and y in square domain)
x2 = np.linspace(-L/2, L/2, m + 1)  # m+1 points to cover domain from -L/2 to L/2
x = x2[:m]  # Exclude the last point to maintain periodic boundaries

n = m * m         # Total matrix size

#----------------------------------MATRIX A------------------------------------
# Generate matrix A for ∂²x + ∂²y with periodic boundaries
e0 = np.zeros((n, 1))
e1 = np.ones((n, 1))
e2 = np.copy(e1)
e4 = np.copy(e0)

for j in range(1, m+1):
    e2[m * j - 1] = 0   # overwrite every mth value with zero
    e4[m * j - 1] = 1   # overwrite every mth value with one

# Shift vectors for periodic boundaries
e3 = np.zeros_like(e2)
e3[1:n] = e2[0:n-1]
e3[0] = e2[n-1]

e5 = np.zeros_like(e4)
e5[1:n] = e4[0:n-1]
e5[0] = e4[n-1]

# Define diagonals and offsets for the Laplacian matrix A
diagonals_A = [
    e1.flatten(), e1.flatten(), e5.flatten(), e2.flatten(),
    -4 * e1.flatten(), e3.flatten(), e4.flatten(), e1.flatten(), e1.flatten()
]
offsets_A = [-(n-m), -m, -m+1, -1, 0, 1, m-1, m, (n-m)]

A1_matrix = spdiags(diagonals_A, offsets_A, n, n).toarray() / (dx ** 2)

A1_matrix[0,0]=2/dx**2

#----------------------------------MATRIX B------------------------------------

# PART B: Matrix B (A2) - ∂x derivative with periodic boundaries
diagonals = [e1.flatten(), -e1.flatten(), e1.flatten(), -e1.flatten()]
offsets = [-(n-m), -m, m, (n-m)]
A2_matrix = spdiags(diagonals, offsets, n, n).toarray() / (2 * dx)

#----------------------------------MATRIX C------------------------------------

# PART C: Matrix C (A3) - ∂y derivative with periodic boundaries
# Define e1, e2, e3, e4 according to the professor's instructions
e1 = np.tile([1, 0, 0, 0, 0, 0, 0, 0], m)
e2 = np.tile([1, 1, 1, 1, 1, 1, 1, 0], m)
e3 = np.tile([0, 1, 1, 1, 1, 1, 1, 1], m)
e4 = np.tile([0, 0, 0, 0, 0, 0, 0, 1], m)

val_C = [e1, -e2, e3, -e4]
offset_C = [-(m-1), -1, 1, m-1]
A3_matrix = spdiags(val_C, offset_C, n, n).toarray() / (2 * dx)

"""
#------------------------------------------------------------------------------
"""

# Step 5: Define the PDE
def pde(t, w2, nx, ny, N, K, A1_matrix, A2_matrix, A3_matrix, nu):
    
    print(np.shape(w2))
    # Reshape w to 2D for FFT operations
    w = w2.reshape((nx, ny))
    
    # Compute Fourier Transform of w
    wt = fft2(w)
    
    # Solve for ψ in spectral space (∇²ψ = ω → ψ = -ω/K)
    psit = -wt / K
    
    # Transform ψ back to real space
    psi = np.real(ifft2(psit)).reshape(N)
    
    # Derivatives using matrices
    #Bpsi = A2_matrix @ psi  # ∂ψ/∂x
    #Cpsi = A3_matrix @ psi  # ∂ψ/∂y
    #Bw = A2_matrix @ w2          # ∂w/∂x
    #Cw = A3_matrix @ w2          # ∂w/∂y
    
    # Derivatives using matrices
    Bpsi = np.dot(A2_matrix,psi)  # ∂ψ/∂x
    Cpsi = np.dot(A3_matrix,psi)  # ∂ψ/∂y
    Bw = np.dot(A2_matrix,w2)          # ∂w/∂x
    Cw = np.dot(A3_matrix, w2)          # ∂w/∂y
    
    
    # Compute RHS
    rhs = nu * np.dot(A1_matrix,w2) - (Bpsi * Cw) + (Cpsi * Bw)
    
    return rhs

# Step 6: Solve the PDE using solve_ivp
solution = solve_ivp(
    pde,                        
    [0, 4],                          
    w2,                             
    t_eval=tspan,                    
    args=(nx, ny, N, K, A1_matrix, A2_matrix, A3_matrix, nu), 
    method='RK45'                   
)

# Extract solution
A1 = solution.y

# Output for verification
print("Shape of A1:", A1.shape)  # Should be (4096, 9)
print("A1 matrix:", A1)


# Determine the number of rows and columns for subplots
num_plots = len(tspan)
rows = (num_plots + 2) // 3  # Adjust rows to fit all subplots in 3 columns

# Graphing the solution
plt.figure(figsize=(12, 8))
for j, t in enumerate(solution.t):
    w = solution.y[:, j].reshape((nx, ny))  # Reshape the solution for the j-th time step
    plt.subplot(3, 3, j + 1)
    plt.pcolor(x, y, w, shading='auto')  # Adjusted shading to 'auto'
    plt.title(f'Time: {t:.1f}')
    plt.colorbar()

plt.tight_layout()
plt.show()


