import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

# -----------------------------------------

# Beam parameters

# Young's modulus (Pa)
E = 70e9          

# Diameter (m)
d = 0.005         

# Beam length (m)
L = 0.20          

# -----------------------------------------

# Desired downward displacement of the tip (m)
y_target = -0.12

# -----------------------------------------

# Second moment of the area (m^4)
I = np.pi * d**4 / 64


# Elastica equations
def elastica(s, u, p):
    F = p[0]

    theta = u[0]
    kappa = u[1]

    dx_ds = np.cos(theta)
    dy_ds = np.sin(theta)

    dtheta_ds = kappa
    dkappa_ds = -(F / (E * I)) * np.cos(theta)

    return np.vstack((dtheta_ds, dkappa_ds, dx_ds, dy_ds))


# Boundary conditions
def boundary_conditions(u0, uL, p):
    return np.array([
        # theta(0) = 0
        u0[0],               
        # x(0) = 0
        u0[2],               
        # y(0) = 0
        u0[3],               
        # kappa(L) = 0
        uL[1],               
        # y(L) = desired displacement
        uL[3] - y_target     
    ])


# Initial placeholder guesses
s = np.linspace(0, L, 100)

theta_guess = np.linspace(0, 0.5, 100) # linear from 0-0.5 deg
kappa_guess = np.gradient(theta_guess, s) # derivative of angle 
x_guess = s # same as euler-bernoulli
y_guess = y_target * (s / L) # linear to known end displacement

u_guess = np.vstack((
    theta_guess,
    kappa_guess,
    x_guess,
    y_guess
))

# Initial guess for the force
F_guess = 1.0

solution = solve_bvp(
    elastica,
    boundary_conditions,
    s,
    u_guess,
    p=[F_guess],
    tol=1e-6,
    max_nodes=10000
)

if not solution.success:
    raise RuntimeError("BVP solver did not converge.")

# Evaluate solution
s_plot = np.linspace(0, L, 300)
u = solution.sol(s_plot)

theta = u[0]
x = u[2]
y = u[3]

F = solution.p[0]

print(f"Required tip force: {F:.3f} N")
print(f"Tip displacement:    {y[-1]:.5f} m")
print(f"Tip x-position:      {x[-1]:.5f} m")

# -----------------------------------------

# Plot
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    s_plot,
    np.zeros_like(s_plot),
    '--',
    label='Undeformed'
)

ax.plot(
    x,
    y,
    linewidth=2,
    label='Deformed'
)

ax.scatter(
    x[-1],
    y[-1],
    zorder=3
)

ax.set_xlim(0, L)
ax.set_ylim(-L,0.05 * L)

ax.set_aspect('equal', adjustable='box')

ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_title('Reconstructed Beam Shape')

ax.grid(True)
ax.legend()

plt.show()