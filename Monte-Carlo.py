# Projections de points MC et QMC sur Aire Fonction de Masse

import numpy as np
import scipy.stats as ssp
import matplotlib.pyplot as plt

num_sample = 1000
U = np.random.uniform(size=num_sample)
S = ssp.qmc.Sobol(d=1).random(num_sample).reshape(num_sample)
n = 1200
p = 0.25
x = 270

def Bino(u, n=n, p=p):
    proba = np.array([ssp.binom.pmf(k, n=n, p=p) for k in range(n + 1)])

    return np.searchsorted(np.cumsum(proba), u)


Bino_alea = Bino(U)
Bino_non_alea = Bino(S)

Proba_Monte_Carlo = sum(Bino_alea <= x) / num_sample
Proba_Quasi_Monte_Carlo = sum(Bino_non_alea <= x) / num_sample
Proba_numpy = sum(np.random.binomial(n=n, p=p, size=num_sample) <= x) / num_sample
Vraie_Proba = ssp.binom.cdf(x, n=n, p=p)

print("Proba MC = {PMC}".format(PMC=Proba_Monte_Carlo))
print("Proba QMC = {PQMC}".format(PQMC=Proba_Quasi_Monte_Carlo))
print("Proba Np = {Np}".format(Np=Proba_numpy))
print("Vraie  = {VP}".format(VP=Vraie_Proba))


# MC Uniforme
import numpy as np

num_sample = 100000
n = 1200
total = []

for i in range(num_sample):
    s = np.random.uniform(size=1200)
    resultat = []
    for element in s:
        if 0 <= element < 0.25:
            resultat.append(1)
        elif 0.25 <= element < 1:
            resultat.append(0)

    nb_de_1 = resultat.count(1)
    if nb_de_1 < 270:
        total.append(1)

print(len(total) / num_sample)


# QMC avec Sobol
import scipy.stats as ssp
import numpy as np
import random

num_sample = 100
n = 1200
total = []

for i in range(num_sample):
    s =  [ssp.qmc.Sobol(d=1).random(1) for i in range(1200)]
    resultat = []
    for element in s:
        if 0 <= element < 0.25:
            resultat.append(1)
        elif 0.25 <= element < 1:
            resultat.append(0)

    nb_de_1 = resultat.count(1)
    if nb_de_1 < 270:
        total.append(1)

print(len(total) / num_sample)

# Estimation intégrale avec MC et QMC

import scipy.stats as ssp
import numpy as np
import random

# Intégrale avec une fonction ssp
def f_u(u):
    return np.cos(u)**2 * np.exp(np.sin(u**2)) * np.tan(u)
import scipy.integrate as integrate
import scipy.special as special

test = integrate.quad(f_u, 0, 1)
test


# Intégrale avec MC
import scipy.stats as ssp
import numpy as np
import random
def f(u):
    return np.cos(u)**2 * np.exp(np.sin(u**2)) * np.tan(u)

def monte_carlo(f, a, b, n_iterations):
    somme = 0
    for i in range(n_iterations):
        x = random.uniform(a, b)
        somme += f(x)
    integrale = (b - a) * (somme / n_iterations)
    return integrale

resultat = monte_carlo(f, 0, 1, 10000)
print(resultat)

# Intégrale avec QMC

import scipy.stats as ssp
import numpy as np
import random
from tqdm import tqdm
def f(u):
    return np.cos(u)**2 * np.exp(np.sin(u**2)) * np.tan(u)

def monte_carlo_sobol(f, a, b, n_iterations):
    somme = 0
    for i in tqdm(range(n_iterations)):
        x = ssp.qmc.Sobol(d=1).random(n_iterations)[i]
        somme += f(x)
    integrale = (b - a) * (somme / n_iterations)
    return integrale

resultat = monte_carlo_sobol(f, 0, 1, 10000)
print(resultat)

# Intégrale avec une suite random
def f(u):
    return np.cos(u)**2 * np.exp(np.sin(u**2)) * np.tan(u)
def monte_carlo_random(f, a, b, n_iterations):
    somme = 0
    for i in range(n_iterations):
        x = [number*0.0001 for number in range(10000)]
        somme += f(x[i])
    integrale = (b - a) * (somme / n_iterations)
    return integrale

resultat = monte_carlo_random(f, 0, 1, 10000)
print(resultat)


# TCL
import numpy as np

results = []
n = 1000000
for i in tqdm(range(n)):
    x = np.random.binomial(1,0.5,10)
    results.append(sum(x==1))

plt.hist(results)
plt.show()


sl = [1,2,2,3,3,3,4,4,4,4,3]

