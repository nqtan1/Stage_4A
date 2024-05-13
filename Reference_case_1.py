import numpy as np
from scipy.linalg import eigh

def reference_case_1(nb_elements, nb_freq, precision, instance):
    # Define material and geometry parameters
    Eyoung = 2e11  # Pa, steel
    L_tot = 1.0    # m
    rho = 7850     # kg/m^3, density
    h = 0.0053     # m
    b = 0.0249     # m
    S = b * h      # m^2, surface
    I = (b * h ** 3) / 12  # m^4, quadratic moment
    L = L_tot / nb_elements  # m, partition length

    # Create an array for the instance (example random values)
    x = instance

    # Define elementary matrices
    E = Eyoung * (1 - x)

    Ke2 = (I / L ** 3) * np.array([[12, 6 * L, -12, 6 * L],
                                  [6 * L, 4 * L ** 2, -6 * L, 2 * L ** 2],
                                  [-12, -6 * L, 12, -6 * L],
                                  [6 * L, 2 * L ** 2, -6 * L, 4 * L ** 2]])

    Me = ((rho * S * L) / 420) * np.array([[156, 22 * L, 54, -13 * L],
                                          [22 * L, 4 * L ** 2, 13 * L, -3 * L ** 2],
                                          [54, 13 * L, 156, -22 * L],
                                          [-13 * L, -3 * L ** 2, -22 * L, 4 * L ** 2]])

    size_ke_total = 2 * (nb_elements + 1)

    K_rigidite = np.zeros((size_ke_total, size_ke_total))
    M_masse = np.zeros((size_ke_total, size_ke_total))

    # Assembly
    for j in range(nb_elements):
        Ke = E[j] * Ke2
        for k in range(4):
            for n in range(4):
                K_rigidite[n + 2 * j, k + 2 * j] += Ke[n, k]
                M_masse[n + 2 * j, k + 2 * j] += Me[n, k]

    # Free imbed beam
    K_rigidite = K_rigidite[2:, 2:]
    M_masse = M_masse[2:, 2:]

    # Modal analysis
    Val_prop, Vect_prop = eigh(K_rigidite, M_masse)
    Val_prop = np.real(Val_prop)
    Vect_prop = np.real(Vect_prop)

    Val_prop_range = np.sort(np.diag(Val_prop))
    Puls_prop = np.sqrt(Val_prop_range)
    freq_hz = Puls_prop / (2 * np.pi)
    freq_hz = freq_hz[:nb_freq]

    freq = np.round(freq_hz, precision)
    indices_nonzero=np.nonzero(freq)
    freq=freq[indices_nonzero]

    return x, freq