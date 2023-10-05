import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import xlwt
from xlwt import Workbook

# Curvas e parâmetros

df_gss = pd.read_excel(r'C:\Users\Vinicius\Documents\Vinícius\Faculdade\Iniciação Científica\Dados\Boxplots\Boxplot AC\Gaussiana\gaussIM.xlsx')

meses = df_gss['Meses'].values.tolist()
coleta = df_gss['Coleta'].values.tolist()
x = meses
y = coleta

def gauss_f(x, A, mu, sigma):
    return A * np.exp(-(x-mu)**2 / sigma**2)

popt, pcov = curve_fit(gauss_f, x , y)

A_opt, mu_opt, sigma_opt = popt
x_model = np.linspace(min(x), max(x), 365)
y_model = gauss_f(x_model, A_opt, mu_opt, sigma_opt)

max_y = max(y_model)
y_model_norm = []

for i in y_model:
    j = i/max_y
    y_model_norm.append(j)

norm_crr_1 = []
for i in y_model_norm:
    j = i - 0.369
    norm_crr_1.append(j)

norm_crr_2 = []
for i in norm_crr_1:
    norm_crr_2.append(i)
del norm_crr_2[205:]
for i in range(205,365):
    norm_crr_2.append(0)
    
y_model_ref = []
for i in norm_crr_2:
    y_model_ref.append(i)
del y_model_ref[365:]
for i in norm_crr_2:
    y_model_ref.append(i)
    

X = np.arange(0,365)

X_shift = []
for i in X:
    j = i + 0
    X_shift.append(j)



def função_sazonal_ovp(frames):
    taxa_ovp = 0.4    
    Y = y_model_ref
    ovp_list = []

    nova_taxa_ovp = taxa_ovp * Y[frames]
    ovp_list.append(nova_taxa_ovp)
                
    return ovp_list

def função_sazonal_mort(frames):
    taxa_mortalidade_IM = 0.2
    Y = y_model_ref
    mu_list = []
    if frames < 719:
        nova_taxa_mort = taxa_mortalidade_IM * Y[frames]
        mu_list.append(nova_taxa_mort)
    else:
        nova_taxa_mort = 0
        mu_list.append(nova_taxa_mort)

    return mu_list


frames = 365
ovp_list = []
for i in range(0, frames):
    ovp_list.append(função_sazonal_ovp(i))
mu_list = []
for i in range(0, frames):
    mu_list.append(função_sazonal_mort(i))


zeros_ovp = []
for i in range(0,160):
    zeros_ovp.append(ovp_list[230])
ovp_shift = ovp_list[160:]
for i in zeros_ovp:
    ovp_shift.append(i)


zeros_mu = []
for i in range(0,85):
    zeros_mu.append(ovp_list[230])
mu_pk = mu_list[0:280]

mu_shift = []
for i in zeros_mu:
    mu_shift.append(i)
for i in mu_pk:
    mu_shift.append(i)


plt.plot(X, ovp_shift, color = 'r', label = 'oviposição')
plt.plot(X, mu_shift, color = 'b', label = 'mortalidade imaturos')
#plt.gca().legend(('oviposição','mort. imaturos'))
#plt.title('Curvas Sobrepostas')
#plt.show()





