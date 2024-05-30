import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time
from aim_fcn import AimFcn

# PSO parameters and initial settings
nbElements = 30
BoundaryMin = 0
BoundaryMax = 1
Boundary = np.tile([BoundaryMin, BoundaryMax], (nbElements, 1))
nbMaxPSO = 1
Maxiter = 500
Npar = 200
c1 = 2.05
c2 = 2.05
precision = -4
Wmax = 0.95
Wmin = 0.4

# Pré-allocation
population = np.zeros((nbMaxPSO, nbElements))
tempsdecalcul = np.zeros(nbMaxPSO)
fitGlobal = np.zeros(nbMaxPSO)
scatLimit  = 0.02

# Initializing display
taille_defaut = 10
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.set_size_inches(10, 8)

plt.xlabel('Elements')
plt.ylabel('Fitness')
plt.title('PSO Algo')

ax1.set_xlim(0, 1)
ax1.set_ylim(-0.08, 0.1)


# Draw main rectangles
ax1.add_patch(Rectangle((0, 0), 1, 0.1, linewidth=2, edgecolor='black', facecolor='white'))
ax1.add_patch(Rectangle((0, 0), 0.006, 0.2, linewidth=2, edgecolor='black', facecolor='black'))

R = []

# Draw small rectangles and add text below them
for ii in range(nbElements):
    ax1.add_patch(Rectangle((ii / nbElements, 0), 0.01, 0.1, linewidth=0.002, edgecolor='black', facecolor='black'))
    ax1.text((ii + 0.5) / nbElements, -0.02, str(ii + 1), fontsize=8, ha='center')  # Adjusted position
    R.append(Rectangle((0, 0), 0, 0, linewidth=0.002, edgecolor='black', facecolor='white'))

# Display the last number if nbElements > 10
if nbElements > 10:
    ax1.text((nbElements + 0.5) / nbElements, -0.02, str(nbElements + 1), fontsize=8, ha='center')
else:
    ax1.text((nbElements + 0.5) / nbElements, -0.02, str(nbElements + 1), fontsize=8, ha='center')

ax1.text(0.05, -0.04, 'Fitness : ')
t = ax1.text(0.2, -0.04, 'inf')
ax1.text(0.05, -0.06, 'Statut : ')
statut = ax1.text(0.2, -0.06, 'En cours')
plt.ion()  # Turn on interactive mode
plt.show()

# PSO Loop
for nbPSO in range(nbMaxPSO):
    start_time = time.time()
    X = np.zeros((Npar, nbElements))
    Pbest = np.zeros((Npar, nbElements))
    V = np.zeros((Npar, nbElements))
    Vmax = np.ones(nbElements)
    Gbest = np.zeros(nbElements)
    Gbest_old = 1E-02
    Fitness = np.zeros(Npar)
    Fitness_old = np.ones(Npar) * 1E+02

    # Initialize Pbest, velocity V, and Position X
    for i in range(Npar):
        for j in range(nbElements):
            Pbest[i, j] = np.random.rand() * nbElements
            X[i, j] = BoundaryMin + np.random.rand() * (BoundaryMax - BoundaryMin) * nbElements
            V[i, j] = np.random.rand() * nbElements
        if AimFcn(Pbest[i, :]) < Gbest_old:
            Gbest = Pbest[i, :]
            Gbest_old = AimFcn(Gbest)

    # PSO iterations
    for generation in range(Maxiter):
        W = Wmax - ((Wmax - Wmin) * generation) / Maxiter
        for i in range(Npar):
            for j in range(nbElements):
                r1 = np.random.rand()
                r2 = np.random.rand()
                V[i, j] = W * V[i, j] + c1 * r1 * (Pbest[i, j] - X[i, j]) + c2 * r2 * (Gbest[j] - X[i, j])
                if abs(V[i, j]) > Vmax[j]:
                    V[i, j] = np.sign(V[i, j]) * Vmax[j]
                X[i, j] = X[i, j] + V[i, j]
                if X[i, j] > BoundaryMax or X[i, j] < BoundaryMin:
                    X[i, j] = BoundaryMin
            Fitness[i] = AimFcn(X[i, :])
            if Fitness[i] < Fitness_old[i]:
                Pbest[i, :] = X[i, :]
                Fitness_old[i] = Fitness[i]

        Gbest_new, Pos = min(Fitness), np.argmin(Fitness)
       
        if Gbest_new < Gbest_old:
            Gbest_old = Gbest_new
            Gbest = X[Pos, :]
            for i in range(nbElements):
                if Gbest[i] > Gbest_old:
                    R[i].set_height(0.1 * Gbest[i])
                    R[i].set_facecolor('red')  # Set color to red for fitness larger than previous
                else:
                    R[i].set_height(0.1 * Gbest[i])
                    R[i].set_facecolor('white')
        t.set_text(str(min(Fitness)))
        plt.draw()
        plt.pause(0.02)  

        # Scatter plot update
        ax2.clear()
        thetas = 2 * (np.arange(Npar) * np.pi) / Npar
        x = Fitness * np.cos(thetas)
        y = Fitness * np.sin(thetas)
        ax2.scatter(x, y)
        ax2.set_xlim([-scatLimit, scatLimit])
        ax2.set_ylim([-scatLimit, scatLimit])
        plt.draw()
        plt.pause(0.02)  # Pause to update the plot

    population[nbPSO, :] = Gbest
    fitGlobal[nbPSO] = AimFcn(population[nbPSO, :])
    if fitGlobal[nbPSO] == 0:
        break
    tempsdecalcul[nbPSO] = time.time() - start_time
    print(f"PSO {nbPSO+1}: Fitness={fitGlobal[nbPSO]:.4f}, Time={tempsdecalcul[nbPSO]:.2f}s")

# Final display update
statut.set_text('Terminé')
plt.draw()
plt.pause(0.02)  # Ensure the final update is drawn

# Results
b, bi = min(fitGlobal), np.argmin(fitGlobal)
sol = population[bi, :]
print(f"Best solution: {sol}")
print(f"Best fitness: {AimFcn(sol)}")
plt.ioff()  # Turn off interactive mode
plt.show()
