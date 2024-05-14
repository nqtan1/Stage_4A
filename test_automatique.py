from class_data_PSO import dataPSO,listDataPSO
from class_PSO import PSO
import numpy as np
import warnings

# Désactiver temporairement les avertissements RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)

inputData = '/home/nguyenquoctan/Downloads/Stage_INSA/S17/Source_PSO/Data_Test/Lirer/data.txt'
outputData = '/home/nguyenquoctan/Downloads/Stage_INSA/S17/Source_PSO/Data_Test/Ecrire/solutions.txt'

data = listDataPSO()
methode = PSO()

data.read_from_txt(inputData)
maxPSO = 10

for i in range(data.nbCas()):
    print("================================================================================")
    print("La série frequence no." + str(i) +"  :")
    print(np.array(data.data[i].tabFrequences))
    iterMin, defauts , population , fit = methode.enable_PSO(np.array(data.data[i].tabFrequences),maxPSO)
    print("La meilleure solution: ")
    print(population[iterMin])
    solution = dataPSO(nbDefauts=defauts,tabFrequences=np.array(data.data[i].tabFrequences),tabSolutions=np.array(population[iterMin]),precisionFrequence=3,precisionSolution=2)
    solution.write_to_txt(filePath=outputData)