from class_data_PSO import dataPSO,listDataPSO
from class_PSO import PSO
import numpy as np
import warnings
import time 

# Désactiver temporairement les avertissements RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Récupérer les paramètres du clavier
def get_pso_parameters_from_input(input_string):
    params = input_string.split(',')
    params_dict = {}
    for param in params:
        key, value = param.split('=') 
        params_dict[key.strip()] = value.strip() 
        
    c1 = float(params_dict["c1"])
    c2 = float(params_dict["c2"])
    w_min = float(params_dict["w_min"])
    w_max = float(params_dict["w_max"])
    nbElements = int(params_dict["nbElements"])
    MaxIter = int(params_dict["MaxIter"])
    nbPar = int(params_dict["nbPar"])
    nbPSO = int(params_dict["nbPSO"])
    precisionFrequence = int(params_dict["precisionFrequence"])
    precisionSolution = int(params_dict["precisionSolution"])

    return c1, c2, w_min, w_max, nbElements, MaxIter, nbPar, nbPSO, precisionFrequence, precisionSolution

# Initialization
methode = PSO()
inputData = ''
outputData = ''
data = listDataPSO()

# Interagir avec l'utilisateur à l'aide du terminal
print("============================================================================================")
print("=                                   Méthode PSO                                            =")
print("=    Les paramètres par défaut de PSO sont:                                                =")
print("=        1. Coefficients cognitifs et sociaux : c1 = 2 et c2 = 2                           =")
print("=        2. Nombre de particules  : nbPar =  200                                           =")
print("=        3. Nombre d'itérations maximum   : MaxIter =  50                                  =")
print("=        4. Nombre de PSO  : nbPSO =  100                                                  =")
print("=        5. Domaine de variation du poids inertiel   : w_min = 0.4 et w_max = 0.95         =")
print("=        6. Nombre d'éléments  : nbElements =  10                                          =")
print("=        7. Précision de la fréquence et la solution  : precisionFrequence =  4            =")
print("=                                                       precisionSolution = 2              =")
print("============================================================================================")
print("-> Entrez les valeurs de 1 à 7 pour modifier les paramètres PSO")
print("-> Ou 8 pour modifiez-les tous en même temps selon la syntaxe suivante: c1=2,c2=2,nbPar=100,MaxIter=100,nbPSO=100,w_min=0,w_max=1,nbElements=20,precisionFrequence=4,precisionSolution=2")
print("-> N'appuyez pas sur 0 pour continuer et utiliser les valeurs par défaut (Si rien n'a changé avant)")

while True:

    print(methode.__info__()) # Voir les changements de paramètres après chaque opération

    choice = int(input("Entrez votre choix: "))

    if choice == 0 or choice == 9 :
        break
    elif choice == 8:
        input_string = input("Entrez la syntaxe pour modifier tous les paramètres en même temps: ")
        c1, c2, w_min, w_max, nbElements, MaxIter, nbPar,nbPSO,precisionFrequence, precisionSolution = get_pso_parameters_from_input(input_string)
        methode.update_params(c1, c2, w_min, w_max, nbElements, MaxIter, nbPSO , nbPar, precisionFrequence, precisionSolution)
        print("Paramètres PSO mis à jour avec succès!")
    elif choice in range(1, 7):
        if choice == 1:
            value_c1 = float(input(f"Entrez la nouvelle valeur pour c1 : "))
            value_c2 = float(input(f"Entrez la nouvelle valeur pour c2 : "))
            methode.update_c1(value_c1)
            methode.update_c2(value_c2)
        elif choice == 2:
            value_nPar = float(input(f"Entrez la nouvelle valeur pour le nombre de particules : "))
            methode.update_nbPar(value_nPar)
        elif choice == 3:
            value_maxIter = int(input(f"Entrez la nouvelle valeur pour le nombre d'itérations maximum : "))
            methode.update_MaxIter(int(value_maxIter))
        elif choice == 4:
            value_nbPSO = int(input(f"Entrez la nouvelle valeur pour le nombre PSO : "))
            methode.update_nbPSO(value_nbPSO)
        elif choice == 5:
            value_w_min = float(input(f"Entrez la nouvelle valeur pour w_min : "))
            value_w_max = float(input(f"Entrez la nouvelle valeur pour w_max : "))
            methode.update_w_min(value_w_min)
            methode.update_w_max(value_w_max)
        elif choice == 6:
            value_nbElements = int(input(f"Entrez la nouvelle valeur pour le nombre d'éléments  : "))
            methode.update_nbElements(value_nbElements)
        elif choice == 7:
            value_precision_frequence = int(input(f"Entrez la précision de la fréquence : "))
            value_precision_solution = int(input(f"Entrez la précision de la solution : "))
            methode.update_precisionFrequence(value_precision_frequence)
            methode.update_precisionSolution(value_precision_solution)
        print(f"Paramètre mis à jour avec succès!")
    else:
        print("Choix invalide. Veuillez entrer une valeur de 0 à 8!")
        
print("============================================================================================")
print("=                                   Lien de données                                        =")
print("============================================================================================")

'''Pour Linux, Obtenir le chemin absolu d'un fichier: readlink -f 'nom_fichier' '''

inputData = input("Lien d'accès à l'ensemble de données d'entrée: ")
outputData = input("Lien d'accès pour écrire les données de résultat: ")


print("============================================================================================")
print("=                                   Exécuter le programme                                  =")
print("============================================================================================")

data.read_from_txt(inputData)

for i in range(data.nbCas()):
    print("============================================================================================")
    print("La série fréquence no." + str(i) +"  :")
    print(np.array(data.data[i].tabFrequences))
    start_time = time.time()
    iterMin, defauts , population , fit = methode.enable_PSO(np.array(data.data[i].tabFrequences))
    end_time = time.time()
    print("La meilleure solution: ")
    print(population[iterMin])
    solution = dataPSO(nbDefauts=defauts,tabFrequences=np.array(data.data[i].tabFrequences),tabSolutions=np.array(population[iterMin]),precisionFrequence=4,precisionSolution=2)
    solution.write_to_txt(filePath=outputData)
    print("Durée d'exécution du programme: " + str(end_time-start_time))

print("===================================== C'est fini! ===========================================")