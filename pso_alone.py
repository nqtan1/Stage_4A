import numpy as np
from scipy.linalg import eigh
from AimFcn import aim_fcn
import time

def pso_alone(Fref, nbMaxPSO):
    # Nombre d'éléments (Modélisation EF)
    nbElements = 10
    # Paramètres méthode PSO
    Maxiter = 50   # Nbre d'itération (20 si moins cela ne converge pas exactement vers le résultat)
    Npar = 200   # Nbre de particules (dimension du problème = 100)
    c1 = 2    # Cognitive acceleration (2 ou 2.05 ?)
    c2 = 2    # Social acceleration    (2 ou 2.05 ?)
    # Linearly decreasing inertia weight parameters
    Wmax = 0.95 
    Wmin = 0.4
    # Définition des bornes min,max de chaque élément
    BoundaryMin = 0
    BoundaryMax = 1
    Bound_min = BoundaryMin * np.ones((Npar, nbElements))
    Bound_max = BoundaryMax * np.ones((Npar, nbElements))
    # Pré-allocation
    population = np.zeros((nbMaxPSO, nbElements))
    fitGlobal = np.zeros(nbMaxPSO)
    
    # PSO LOOP
    for nbPSO in range(nbMaxPSO):
        # Pré-allocation des variables
        X = np.zeros((Npar, nbElements))
        Pbest = np.zeros((Npar, nbElements))
        V = np.zeros((Npar, nbElements))
        Vmax = np.ones(nbElements)
        Gbest = np.zeros(nbElements)
        Gbest_old = 1E-02 # to be tested
        Fitness = np.zeros(Npar)
        Fitness_old = np.zeros(Npar)
        Fitness_old[:] = 1E+02 # to be tested
        
        # Initialisation des variables Pbest, vitesse V et Position X
        for i in range(Npar):
            for j in range(nbElements):
                X[i, j] = BoundaryMin + np.random.rand() * (BoundaryMax - BoundaryMin)
                V[i, j] = np.random.rand() * nbElements
            Pbest[i, :] = X[i, :]
            # Choix du Gbest de départ
            if aim_fcn(Pbest[i, :], Fref) < Gbest_old:
                Gbest = Pbest[i, :]
                Gbest_old = aim_fcn(Gbest, Fref)
        # ITERATION 
        for generation in range(Maxiter):
            # Linearly decreasing inertia weight
            W = (Wmax - ((Wmax - Wmin) * generation) / Maxiter)
            
            for i in range(Npar):
                for j in range(nbElements):
                    # Calcul de la vitesse
                    r1 = np.random.rand()
                    r2 = np.random.rand()
                    V[i, j] = W * V[i, j] + c1 * r1 * (Pbest[i, j] - X[i, j]) + c2 * r2 * (Gbest[j] - X[i, j])
                    # Test de seuil de la vitesse
                    if abs(V[i, j]) > Vmax[j]:
                        V[i, j] = np.sign(V[i, j]) * Vmax[j]
                    # Affectation de la nouvelle position de la particule
                    X[i, j] = X[i, j] + V[i, j]
            
            L1_X = X > Bound_max
            X[L1_X == 1] = Bound_max[L1_X == 1]
            L2_X = X < Bound_min
            X[L2_X == 1] = Bound_min[L2_X == 1]
            for i in range(Npar):
                # Calcul de la fonction fitness
                Fitness[i] = aim_fcn(X[i, :], Fref)
                # Test et nouvelles affectations
                if Fitness[i] < Fitness_old[i]:
                    Pbest[i, :] = X[i, :]
                    Fitness_old[i] = Fitness[i]
            # Sélection de la meilleure particule de l'essaim
            Gbest_new, Pos = np.min(Fitness), np.argmin(Fitness)
            # Test et affectation de la meilleure position et du seuil
            if Gbest_new < Gbest_old:
                Gbest_old = Gbest_new
                Gbest = X[Pos, :]
        # La meilleur solution de chaque pso est stockée dans population().
        population[nbPSO, :] = Gbest
        fitGlobal[nbPSO] = aim_fcn(population[nbPSO, :], Fref)
        
    return population, fitGlobal
