import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time
from aim_fcn import AimFcn 
import warnings 
# Désactiver temporairement les avertissements RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Taille de l'Essaim particulaire (Modélisation EF)
nbElements = 30

# Définition des bornes min,max de chaque élément
BoundaryMin = 0
BoundaryMax = 1
Boundary = np.tile([BoundaryMin, BoundaryMax], (nbElements, 1))

# Paramètres méthode PSO
nbMaxPSO = 1   # Nbre de PSO à lancer
Maxiter = 1000   # Nbre d'itération
Npar = 20   # Nbre de particules

c1 = 2.05    # Cognitive acceleration
c2 = 2.05    # Social acceleration
precision = -4   # 4 chiffres significatifs ou moins ?
Wmax = 0.95
Wmin = 0.4

# Pré-allocation
population = np.zeros((nbMaxPSO, nbElements))
fitGlobal = np.zeros((nbMaxPSO, 1))

# Initialisation de l'affichage
fig = plt.figure(figsize=(10, 8),num='Optimisation par essaims particulaires')

ax1 = plt.subplot(2, 1, 1)

ax1.set_title('Fitness Distribution')
ax1.set_xlabel('Elements')
ax1.set_ylabel('Fitness')

ax1.set_xlim(0, 1)
ax1.set_ylim(-0.08, 0.15)

# Affichage des particules
ax2 = plt.subplot(2, 1, 2)

# Processuss rechercher du PSO
scatLimit = 0.02
ax2.set_xlim(-scatLimit, scatLimit)
ax2.set_ylim(-scatLimit, scatLimit)
thetas = 2 * (np.arange(Npar) * np.pi) / Npar
x = 0.02 * np.cos(thetas)
y = 0.02 * np.sin(thetas)
scat = ax2.scatter(x, y)

plt.tight_layout() 

# Rectangle principale
ax1.add_patch(Rectangle((0, 0), 1, 0.1, linewidth=3, edgecolor='black', facecolor='white'))

# Le tableau contient des valeurs rectangulaires 
# utilisées pour afficher les valeurs 'fitness' changeantes
tab_Rec = []

rec_width = 1 / nbElements # La longueur par défaut d'un rectangle
rec_height = 0.1 # La largeur par défaut d'un rectangle 
linewidth = 0.002 # 

# Dessiner de petits rectangles et ajouter du texte en dessous
for i in range(nbElements):
    bound_width = rec_width * 0.3  # Black rectangle takes 30% of the element slot width
    element_width = rec_width * 0.7    # Red rectangle takes 70% of the element slot width

    black_rect = Rectangle((i * rec_width, 0), width=bound_width, height=0.1, linewidth=linewidth, edgecolor='black', facecolor='black')
    red_rect = Rectangle((i * rec_width + bound_width, 0), width=element_width, height=rec_height, linewidth=linewidth, edgecolor='black', facecolor='green')

    ax1.add_patch(black_rect)
    ax1.add_patch(red_rect)
    ax1.text((i + 0.5) / nbElements, -0.02, str(i + 1), fontsize=8, ha='center')

    tab_Rec.append(red_rect)

ax1.text(0.05, -0.04, 'Fitness : ')
t = ax1.text(0.2, -0.04, 'inf')
ax1.text(0.05, -0.06, 'Statut : ')
statut = ax1.text(0.2, -0.06, 'En cours')
plt.grid(True)
plt.ion()  # Activer le mode interactif
plt.show()

# PSO LOOP
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

    # Initialisation des variables Pbest, vitesse V et Position X
    for i in range(Npar):
        for j in range(nbElements):
            Pbest[i, j] = np.random.rand() * nbElements
            X[i, j] = BoundaryMin + np.random.rand() * (BoundaryMax - BoundaryMin) * nbElements
            V[i, j] = np.random.rand() * nbElements
        # Choix du Gbest de départ
        if AimFcn(Pbest[i, :]) < Gbest_old:
            Gbest = Pbest[i, :]
            Gbest_old = AimFcn(Gbest)

    # ITERATION
    for generation in range(Maxiter):
        W = (Wmax - ((Wmax - Wmin) * generation) / Maxiter)
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
            # Calcul de la fonction fitness
            Fitness[i] = AimFcn(X[i, :])
            if Fitness[i] < Fitness_old[i]:
                Pbest[i, :] = X[i, :]
                Fitness_old[i] = Fitness[i]
        # Sélection de la meilleure particule de l'essaim
        Gbest_new, Pos = min(Fitness), np.argmin(Fitness)
        if Gbest_new < Gbest_old:
            Gbest_old = Gbest_new
            Gbest = X[Pos, :]
            for i in range(nbElements):
                # Mettre à jour la valeur « fitness »
                height_fitness = 0.1 * Gbest[i]
                tab_Rec[i].set_height(height_fitness)
                if height_fitness > 0:
                    tab_Rec[i].set_facecolor('red')
                else:
                    tab_Rec[i].set_facecolor('white')
            # Afficher la valeur « fitness »
            t.set_text(str(min(Fitness)))
            plt.draw()
            plt.pause(0.02)
        # Mise à jour du nuage de points
        ax2.clear()
        thetas = 2 * (np.arange(Npar) * np.pi) / Npar
        x = Fitness * np.cos(thetas)
        y = Fitness * np.sin(thetas)
        ax2.scatter(x, y)
        ax2.set_xlim([-scatLimit, scatLimit])
        ax2.set_ylim([-scatLimit, scatLimit])
        plt.draw()
        plt.pause(0.02)
    # La meilleur solution de chaque pso est stockée dans population().
    population[nbPSO, :] = Gbest
    fitGlobal[nbPSO] = AimFcn(population[nbPSO, :])

# Mise à jour finale de l'affichage
statut.set_text('Terminé')
plt.draw()
plt.pause(0.02)  # Ensure the final update is drawn

# Résultats
b, bi = min(fitGlobal), np.argmin(fitGlobal)
sol = population[bi, :]
print(f"Best solution: {sol}")
print(f"Best fitness: {AimFcn(sol)}")
plt.ioff() # Désactiver le mode interactif
plt.show()
