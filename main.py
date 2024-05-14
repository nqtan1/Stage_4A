from pso_alone import pso_alone
import numpy as np 
from class_PSO import PSO
import warnings

# Désactiver temporairement les avertissements RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Donnés des fréquences
fre0 = np.array([4.3215,27.0834,75.8511,148.7416,246.2656,368.9339,517.6322,693.4844])
fre1 = np.array([4.3149,26.3712,70.7532,141.3329,242.0010,361.2523,499.5734,672.7648])
fre2 = np.array([3.7831,26.5802,73.1423,141.2403,234.4326,347.9824,488.3618,660.0362])
'''
fre3 = np.array([3.9608,25.8035,72.5151,144.0284,232.3650,349.1331,499.1010,664.0319])
fre4 = np.array([3.9405,25.4631,68.2683,138.9173,230.1381,343.9292,482.9024,640.2499])
fre5 = np.array([3.8607,23.4129,70.9617,136.5809,230.6408,344.5962,475.9716,637.3479])
fre5_new = np.array([4.0553,24.0445,68.8111,132.4832,228.3213,336.4345,465.2133,616.1900])

# Utiliser la fonction PSO_alone
pop, fit = pso_alone(fre5_new,300);

'''

# Utiliser la classe PSO
init_PSO = PSO()
print(init_PSO.__info__())
ite,defauts, pop, fit = init_PSO.enable_PSO(fre1)

# Afficher les résultats de l'algorithme PSO
print("======================================================================")
print("Nombre de defauts : " + str(defauts))
#print("Population: ")
#print(pop.view())

print("======================================================================")
print("Fitness Value:")
print(fit.view())

print("======================================================================")
print("Min Fitness Value: " + str(fit.min()))
print("Population: ")
print(pop[ite].view())