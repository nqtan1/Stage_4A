import os

class dataPSO: 
    def __init__(self,nbDefauts=-1,nbFrequences=0,nbElements=0,tabFrequences=[],tabSolutions=[],precisionFrequence=4,precisionSolution=2):
        self.tabFrequences = tabFrequences
        self.tabSolutions = tabSolutions
        self.precisionFrequence = precisionFrequence
        self.precisionSolution = precisionSolution
        
        if len(self.tabSolutions) != 0 and nbElements == 0:
            self.nbElements = len(self.tabSolutions)
        else :
            self.nbElements = nbElements
            
        if len(self.tabFrequences) != 0 and nbFrequences == 0:
            self.nbFrequences = len(self.tabFrequences)
        else :
            self.nbFrequences = nbFrequences
        
        if (self.getNbDefauts()-1) != -1:
            self.nbDefauts = self.getNbDefauts()
        else:
            self.nbDefauts = nbDefauts
            
    def __info__(self):
        info = f"Nombre de défauts: {self.nbDefauts}\n"
        info += f"Nombre de fréquences: {self.nbFrequences}\n"
        info += f"Nombre d'éléments: {self.nbElements}\n"
        info += f"Tableau des fréquences: {self.tabFrequences}\n"
        info += f"Tableau de solutions: {self.tabSolutions}\n"
        info += f"Précision pour les fréquences: {self.precisionFrequence}\n"
        info += f"Précision pour la solution: {self.precisionSolution}\n"
        return info
    
    def getNbDefauts (self):
        count = 0
        i = 0 
        for ite in self.tabSolutions:
            i += 1
            if ite != 0 : 
                count = count + 1
        return count
    
    def string_frequence_for_txt(self):
        text = ''
        for i in range(self.nbFrequences):
            text += f"{self.tabFrequences[i]:.{self.precisionFrequence}f}"
            if self.tabFrequences[i] != self.tabFrequences[-1]:
                text = text + ','
        return text
    
    def string_solution_for_txt(self):
        text = ''
        for i in range(self.nbElements):
            text += f"{self.tabSolutions[i]:.{self.precisionSolution}f}"
            if i != len(self.tabSolutions)-1:
                text = text + ','
        return text
              
    def string_for_txt(self):
        stringFrequences = self.string_frequence_for_txt()
        stringSolutions = self.string_solution_for_txt()
        return str(self.nbDefauts) + "\t" + stringFrequences + "\t" + stringSolutions
    
    def write_to_txt(self,filePath):
        file = open(filePath,'a')
        # Check if the file is empty
        is_empty = os.path.getsize(filePath) == 0
        # Write headers if the file is empty
        if is_empty:        
            file.write(f"	DATASET -> nb_elements : {self.nbElements}; nb_frequences : {self.nbFrequences}; Precision frequence : {self.precisionFrequence} \n")

        file.write(self.string_for_txt()+'\n')
        file.close()
              
class listDataPSO:
    def __init__(self):
        self.data = []
        
    def __info__(self):
        for data in self.data:
            print( data.__info__())
    
    # Renvoie le nombre de cas
    def nbCas (self):
        return len(self.data)
    
    # Lire les données du fichier .txt
    def read_from_txt(self, file_path):
        with open(file_path, 'r') as file:
            # Lire la valeur de l'en-tête
            metadata_line = file.readline().strip()
            metadata_parts = metadata_line.split('; ')
            nb_elements = int(metadata_parts[0].split(': ')[1])
            nb_frequences = int(metadata_parts[1].split(': ')[1])
            precision_frequence = int(metadata_parts[2].split(': ')[1])

            # Liser chaque ligne suivante et créer les objets dataPSO correspondants
            data_objects = []
            for line in file:
                parts = line.strip().split('\t')
                nb_defaut = float(parts[0])
                tab_values_fre = parts[1].split(',')
                tab_values_sol = parts[2].split(',')
                tab_frequence = [float(value) for value in tab_values_fre[:nb_frequences]]
                tab_solution = [float(value) for value in tab_values_sol[:nb_elements]]
                data_objects.append(dataPSO(nbFrequences=nb_frequences, nbElements=nb_elements, tabFrequences=tab_frequence, tabSolutions=tab_solution, precisionFrequence=precision_frequence))
        
        self.data = data_objects