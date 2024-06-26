import numpy as np
from scipy.linalg import eigh
from AimFcn import aim_fcn
from class_data_PSO import dataPSO, listDataPSO
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

 # Renvoie le nombre de défauts
def getNbDefauts (tab,precisionSolution=4):
    # Définir un seuil pour déterminer si une valeur est considérée comme « différente de zéro »
    threshold = 0.01 ** precisionSolution  
    count = 0
    for item in tab:
        # Utiliser abs() si vous souhaitez également prendre en compte les valeurs négatives
        if abs(item) > threshold:  
            count += 1
    return count

def pso_alone_manuel(Fref=[],nbFreq = 8,c1=2,c2=2,Maxiter=50,Wmax=0.95,Wmin=0.4,nbElements=10,Npar=200,nbMaxPSO=50,precisionFrequence=4,precisionSolution=2,outputData=''):
    # Définition des bornes min,max de chaque élément
    BoundaryMin = 0
    BoundaryMax = 1
    Bound_min = BoundaryMin * np.ones((Npar, nbElements))
    Bound_max = BoundaryMax * np.ones((Npar, nbElements))
    # Pré-allocation
    population = np.zeros((nbMaxPSO, nbElements))
    fitGlobal = np.zeros(nbMaxPSO)

    # Ensemble de valeurs après arrondi
    Fref = np.array([round(freq, precisionFrequence) for freq in Fref[:nbFreq]])

    #print("Série de fréquences no." + str(i) + ": ")
    print(Fref)

    start_time = time.time()  

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
                    #print("Position " + str(i) + ": ")
                    #print(X)
            
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

    end_time = time.time()
    running_time = (end_time-start_time) # Temps moyen d'exécution de la recherche du PSO
    print("Temps: " + str(round(running_time,3)) + "s")

    # Retourner le nombre de défauts
    defauts = getNbDefauts(population[iterMin(fitGlobal)],precisionSolution)
    print("Nombre de défaut(s): " +str(defauts))
    itMin = iterMin(fitGlobal) # Emplacement de la valeur minimal

    print("Meilleure solution: ")
    print(population[itMin])
    print("Fitness Value: ")
    print(fitGlobal.min())
    print("======================================================================")

    if outputData == '':
        print("Le chemin du fichier de résultat est introuvable!")
    else:
        solution = dataPSO(nbDefauts=defauts,tabFrequences=Fref,tabSolutions=np.array(population[itMin]),precisionFrequence=precisionFrequence,precisionSolution=precisionSolution)
        solution.write_to_txt(filePath=outputData)
    
    return itMin, running_time , population, fitGlobal


def pso_alone_automatique(c1=2,c2=2,Maxiter=50,Wmax=0.95,Wmin=0.4,nbElements=10,Npar=200,nbMaxPSO=50,nbFreq=8,precisionFrequence=4,precisionSolution=2,inputData='',outputData=''):
    # Définition des bornes min,max de chaque élément
    BoundaryMin = 0
    BoundaryMax = 1
    Bound_min = BoundaryMin * np.ones((Npar, nbElements))
    Bound_max = BoundaryMax * np.ones((Npar, nbElements))
    # Pré-allocation
    population = np.zeros((nbMaxPSO, nbElements))
    fitGlobal = np.zeros(nbMaxPSO)

    # 
    data = listDataPSO()
    data.read_from_txt(inputData)
    
    for i in range(data.nbCas()):
        Fref = np.array(data.data[i].tabFrequences[:nbFreq])
        # Ensemble de valeurs après arrondi
        Fref = np.array([round(freq, precisionFrequence) for freq in Fref])
        print("Série de fréquences no." + str(i) + ": ")
        print(Fref)
        start_time = time.time()  

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
                        #print("Position " + str(i) + ": ")
                        #print(X)
                
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

        end_time = time.time()
        running_time = (end_time-start_time) # Temps moyen d'exécution de la recherche du PSO
        print("Temps: " + str(round(running_time,3)) + "s")

        # Retourner le nombre de défauts
        defauts = getNbDefauts(population[iterMin(fitGlobal)],precisionSolution)
        print("Nombre de défaut(s): " +str(defauts))
        itMin = iterMin(fitGlobal) # Emplacement de la valeur minimal

        print("Meilleure solution: ")
        print(population[itMin])
        print("======================================================================")

        solution = dataPSO(nbDefauts=defauts,tabFrequences=Fref,tabSolutions=np.array(population[itMin]),precisionFrequence=precisionFrequence,precisionSolution=precisionSolution)
        solution.write_to_txt(filePath=outputData)

# Function d’évaluation des performances des algorithmes.
def eval_algo_manuel(solution_test=[], solution_predict=[], epsilon=0.05):
    #Liste contient les valeurs de différence entre le résultat prédit et le résultat réel connu
    tab_eval = []
    for i in range(len(solution_test)):
        tab_eval.append(solution_predict[i] - solution_test[i])

    eval_sol = True
    for value in tab_eval:
        if abs(value) >= epsilon:
            eval_sol = False
            break
    eval_result = {
        "tab_eval": tab_eval,
        "eval_sol": eval_sol
    }

    return eval_result

# Function d’évaluation des performances des algorithmes.
def eval_algo_automatique(solutions_test='',solutions_predict='',epsilon=0.05):
    sol_test = listDataPSO()
    sol_pred = listDataPSO()

    sol_test.read_from_txt(solutions_test)
    sol_pred.read_from_txt(solutions_predict)

    #Liste contient les valeurs de différence entre le résultat prédit et le résultat réel connu
    eval_result = []

    for i in range(sol_test.nbCas()):
        # Lirer la valeur du résultat à partir du fichier .txt
        sol_test_val = np.array(sol_test.data[i].tabSolutions)
        sol_pred_val = np.array(sol_pred.data[i].tabSolutions)

        eval_result.append(eval_algo_manuel(sol_test_val,sol_pred_val,epsilon=epsilon))
        
    return eval_result