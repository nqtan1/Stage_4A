import numpy as np
from scipy.linalg import eigh
from Reference_case_1 import reference_case_1  # Importer la fonction Reference_case_1 depuis le script correspondant

def aim_fcn(x, Fref):
    # Nombre de chiffres significatifs
    precision = 4
    nb_capteur = Fref.size if Fref.ndim == 1 else Fref.shape[1]  # Nombre de capteurs

    nbre_deg_noeud = 2
    nb_elements = nb_elements = x.shape[1] if len(x.shape) > 1 else x.size
    nb_noeuds = nb_elements + 1  # Nombre de noeuds
    Eyoung = 2E11  # Pa ou N/m2

    # Caractéristiques poutres
    L_tot = 1  # m
    rho = 7850  # kg/m3
    h = 0.0053  # m
    b = 0.0249  # m
    S = b * h  # m2
    I = (b * h**3) / 12  # m4
    L = L_tot / nb_elements  # m longueur d'un élément

    # Matrices élémentaires
    E = Eyoung * (1 - x.flatten())
    

    # Raideur élémentaire
    Ke2 = (I / L**3) * np.array([[12, 6 * L, -12, 6 * L],
                                 [6 * L, 4 * L**2, -6 * L, 2 * L**2],
                                 [-12, -6 * L, 12, -6 * L],
                                 [6 * L, 2 * L**2, -6 * L, 4 * L**2]])
    # Masse élémentaire
    Me = ((rho * S * L) / 420) * np.array([[156, 22 * L, 54, -13 * L],
                                           [22 * L, 4 * L**2, 13 * L, -3 * L**2],
                                           [54, 13 * L, 156, -22 * L],
                                           [-13 * L, -3 * L**2, -22 * L, 4 * L**2]])
    sizeKe_totale = nbre_deg_noeud * nb_noeuds
    K_rigidite = np.zeros((sizeKe_totale, sizeKe_totale))
    M_masse = np.zeros((sizeKe_totale, sizeKe_totale))

    # Assemblage
    for j in range(1, nb_elements + 1):
        Ke = E[j - 1] * Ke2
        Ke_totale = np.zeros((sizeKe_totale, sizeKe_totale))
        Me_totale = np.zeros((sizeKe_totale, sizeKe_totale))
        for k in range(1, 5):
            for n in range(1, 5):
                Ke_totale[n + (j - 1) * nbre_deg_noeud - 1, k + (j - 1) * nbre_deg_noeud - 1] = Ke[n - 1, k - 1]
                Me_totale[n + (j - 1) * nbre_deg_noeud - 1, k + (j - 1) * nbre_deg_noeud - 1] = Me[n - 1, k - 1]

        K_rigidite += Ke_totale
        M_masse += Me_totale
    
    # Poutre encastrée-libre

    K_rigidite = np.delete(K_rigidite, np.s_[:2], axis=0)
    K_rigidite = np.delete(K_rigidite, np.s_[:2], axis=1)
    M_masse = np.delete(M_masse, np.s_[:2], axis=0)
    M_masse = np.delete(M_masse, np.s_[:2], axis=1)
    
    # Analyse modale

    Val_prop, Vect_prop = eigh(K_rigidite, M_masse)
    Val_prop = np.real(Val_prop)
    Vect_prop = np.real(Vect_prop)
    Val_prop_Range = np.sort(Val_prop)
    Indice = np.argsort(Val_prop)
    Vect_prop_range = Vect_prop[:, Indice]
    K_range = K_rigidite
    M_range = M_masse
    Puls_prop = np.sqrt(Val_prop_Range)  # Pulsation propre (en rad/s)
    freqHz = Puls_prop / (2 * np.pi)  # UNITS: Hertz
    freqHzz = freqHz[:nb_capteur]
    freqHzz = np.round(freqHzz, precision)
    
    # Fonction Reference_case_1
    _, f_undamage = reference_case_1(nb_elements, nb_capteur, precision, np.zeros(nb_elements))
    delta_f1 = (f_undamage - Fref) / f_undamage
    delta_f2 = (f_undamage - freqHzz) / f_undamage
    delta_f1 = np.squeeze(delta_f1)
    error4_1 = (np.dot(delta_f1.T, delta_f2)**2) / (np.dot(delta_f1.T, delta_f1) * np.dot(delta_f2.T, delta_f2))
    f_new = np.column_stack((freqHzz.T, Fref.T))
    error4_2 = (1 / nb_capteur) * np.sum(np.min(f_new, axis=1) / np.max(f_new, axis=1))
    err = -1 / 2 * (error4_1 + error4_2)
    return err




