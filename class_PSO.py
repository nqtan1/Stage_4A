import numpy as np
from scipy.linalg import eigh
from AimFcn import aim_fcn
import time

# Renvoie l'index de la valeur minimale du tableau
def iterMin (tab):
    iteMin = 0
    valMin = tab[0]
    for i in range(1, len(tab)):
        if tab[i] < valMin:
            valMin = tab[i]
            iteMin = i
    return iteMin
'''
============================================================================================
=                                      Classe PSO                                          =       
=       Propriétés: 1. Coefficients cognitifs et sociaux : c1 et c2                        = 
=                   2. Nombre de particules  : nbPar                                       = 
=                   3. Nombre d'itérations maximum   : MaxIter                             =
=                   4. Nombre de PSO  : nbPSO                                              =  
=                   5. Domaine de variation du poids inertiel   : w_min  et w_max          =
=                   6. Nombre d'éléments  : nbElements                                     =
=       Méthodes:   1. __init__(...) : Déclaration                                         =
=                   2. __info__(): Décrit les informations de la méthode                   =
=                   3. enable_PSO(...) : Exécuter la méthode                               =
=                   4. getNbDefauts() : Renvoie le nombre de défauts                       =
=                   5. update_c1,... : Mettre à jour une ou plusieurs propriétés           =
============================================================================================
'''
class PSO:
    def __init__(self, c1=2, c2=2, w_min=0.4, w_max=0.95, nbElements=10, MaxIter=50, nbPSO=50, nbPar=200 ,precisionFrequence=4,precisionSolution=2):
        self.c1 = c1    # Cognitive acceleration
        self.c2 = c2    # Social acceleration 
        # Linearly decreasing inertia weight parameters
        self.w_min = w_min
        self.w_max = w_max
        self.nbElements = nbElements    # Nombre d'éléments
        self.MaxIter = MaxIter  # Nbre d'itération
        self.nbPar = nbPar   # Nbre de particules 
        self.nbPSO = nbPSO   # Nbre de fois où la recherche a été effectuée
        # Définition des bornes min,max de chaque élément
        self.BoundaryMin = 0
        self.BoundaryMax = 1
        self.Bound_min = self.BoundaryMin * np.ones((self.nbPar, self.nbElements))
        self.Bound_max = self.BoundaryMax * np.ones((self.nbPar, self.nbElements))
        # Nombre de chiffres à arrondir
        self.precisionFrequence = precisionFrequence # pour la fréquence
        self.precisionSolution = precisionSolution   # pour la solution
        self.update_params(c1, c2, w_min, w_max, nbElements, MaxIter, nbPSO, nbPar,precisionSolution,precisionFrequence)
            
    def __info__(self):
        info_str = "PSO :\n"
        info_str += f"c1: {self.c1}\n"
        info_str += f"c2: {self.c2}\n"
        info_str += f"w_min: {self.w_min}\n"
        info_str += f"w_max: {self.w_max}\n"
        info_str += f"nbElements: {self.nbElements}\n"
        info_str += f"MaxIter: {self.MaxIter}\n"
        info_str += f"nbPar: {self.nbPar}\n"
        info_str += f"NbPSO : {self.nbPSO}\n"
        info_str += f"Précision de la fréquence: {self.precisionFrequence}\n"
        info_str += f"Précision de la solution : {self.precisionSolution}\n"
        return info_str

    # Renvoie le nombre de défauts
    def getNbDefauts (self,tab):
        threshold = 0.000001  # Définir un seuil pour déterminer si une valeur est considérée comme « différente de zéro »
        count = 0
        for item in tab:
            if abs(item) > threshold:  # Utiliser abs() si vous souhaitez également prendre en compte les valeurs négatives
                count += 1
        return count
    
    # Exécuter la méthode
    def enable_PSO(self,Fref):
        # Pré-allocation
        population = np.zeros((self.nbPSO, self.nbElements))
        fitGlobal = np.zeros(self.nbPSO)
        
        # PSO LOOP
        for nbPSO in range(self.nbPSO):
            # Pré-allocation des variables
            X = np.zeros((self.nbPar, self.nbElements))
            Pbest = np.zeros((self.nbPar, self.nbElements))
            V = np.zeros((self.nbPar, self.nbElements))
            Vmax = np.ones(self.nbElements)
            Gbest = np.zeros(self.nbElements)
            Gbest_old = 1E-02 # to be tested
            Fitness = np.zeros(self.nbPar)
            Fitness_old = np.zeros(self.nbPar)
            Fitness_old[:] = 1E+02 # to be tested
            
            # Initialisation des variables Pbest, vitesse V et Position X
            for i in range(self.nbPar):
                for j in range(self.nbElements):
                    X[i, j] = self.BoundaryMin + np.random.rand() * (self.BoundaryMax - self.BoundaryMin)
                    V[i, j] = np.random.rand() * self.nbElements
                Pbest[i, :] = X[i, :]
                # Choix du Gbest de départ
                if aim_fcn(Pbest[i, :], Fref) < Gbest_old:
                    Gbest = Pbest[i, :]
                    Gbest_old = aim_fcn(Gbest, Fref)
            # ITERATION 
            for generation in range(self.MaxIter):
                # Linearly decreasing inertia weight
                W = (self.w_max - ((self.w_max  - self.w_min) * generation) / self.MaxIter)
                
                for i in range(self.nbPar):
                    for j in range(self.nbElements):
                        # Calcul de la vitesse
                        r1 = np.random.rand()
                        r2 = np.random.rand()
                        V[i, j] = W * V[i, j] + self.c1 * r1 * (Pbest[i, j] - X[i, j]) + self.c2 * r2 * (Gbest[j] - X[i, j])
                        # Test de seuil de la vitesse
                        if abs(V[i, j]) > Vmax[j]:
                            V[i, j] = np.sign(V[i, j]) * Vmax[j]
                        # Affectation de la nouvelle position de la particule
                        X[i, j] = X[i, j] + V[i, j]
                
                L1_X = X > self.Bound_max
                X[L1_X == 1] = self.Bound_max[L1_X == 1]
                L2_X = X < self.Bound_min
                X[L2_X == 1] = self.Bound_min[L2_X == 1]
                for i in range(self.nbPar):
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
            
            # Retourner le nombre de défauts
            defauts = self.getNbDefauts(population[iterMin(fitGlobal)])
            itMin = iterMin(fitGlobal) # Emplacement de la valeur minimal
            #print("Iter: " +str(itMin))
                        
        return itMin, defauts, population, fitGlobal
    
    # Mettre à jour la valeur de c1
    def update_c1(self, c1):
        self.c1 = c1
    
    # Mettre à jour la valeur de c2
    def update_c2(self, c2):
        self.c2 = c2

    # Mettre à jour la valeur de Wmin
    def update_w_min(self, w_min):
        self.w_min = w_min

    # Mettre à jour la valeur de Wmax
    def update_w_max(self, w_max):
        self.w_max = w_max

     # Mettre à jour la valeur de nbElements
    def update_nbElements(self, nbElements):
        self.nbElements = nbElements

    # Mettre à jour la valeur de MaxIter
    def update_MaxIter(self, MaxIter):
        self.MaxIter = MaxIter

    # Mettre à jour la valeur de Nbre de particules 
    def update_nbPar(self, nbPar):
        self.nbPar = nbPar
        
     # Mettre à jour la valeur de Nbre de PSO  
    def update_nbPSO(self, nbPSO):
        self.nbPSO = nbPSO

    # Mettre à jour la précision de la solution 
    def update_precisionSolution(self, precisionSolution):
        self.precisionSolution = precisionSolution

    # Mettre à jour la précision de la fréquence
    def update_precisionFrequence(self, precisionFrequence):
        self.precisionFrequence = precisionFrequence
    
    # Mettre à jour une ou plusieurs propriétés à la fois avec un seul appel de fonction
    def update_params(self, c1=None, c2=None, w_min=None, w_max=None, nbElements=None, MaxIter=None, nbPSO=None, nbPar=None, precisionSolution=None,precisionFrequence=None):
        if c1 is not None:
            self.update_c1(c1)
        if c2 is not None:
            self.update_c2(c2)
        if w_min is not None:
            self.update_w_min(w_min)
        if w_max is not None:
            self.update_w_max(w_max)
        if nbElements is not None:
            self.update_nbElements(nbElements)
        if MaxIter is not None:
            self.update_MaxIter(MaxIter)
        if nbPar is not None:
            self.update_nbPar(nbPar)
        if nbPSO is not None:
            self.update_nbPSO(nbPSO)
        if precisionFrequence is not None:
            self.update_precisionFrequence(precisionFrequence)
        if precisionSolution is not None:
            self.update_precisionSolution(precisionSolution)
