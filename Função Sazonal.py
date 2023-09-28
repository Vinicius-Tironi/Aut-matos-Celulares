# -*- coding: utf-8 -*-
"""
Created on Tue May 30 19:36:55 2023

@author: Vinicius
"""

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import xlwt
from xlwt import Workbook



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
    

X = np.arange(0,730)

X_shift = []
for i in X:
    j = i + 0
    X_shift.append(j)



plt.plot(X, y_model_ref, color = 'r', label = 'oviposição')
plt.plot(X_shift, y_model_ref, color = 'b', label = 'mortalidade imaturos')
plt.gca().legend(('oviposição','mort. imaturos'))
plt.title('Curvas Sobrepostas')
plt.show()


def função_sazonal_ovp(frames, phi): #, taxa_ovp):
    #taxa_ovp = 0.4
    Y = y_model_ref
    nova_taxa_ovp = phi * Y[frames]
    #print('ovp = ' , Y[frames])
    
    return nova_taxa_ovp


def função_sazonal_mort(frames, mu):  #, taxa_mortalidade_IM):
    #taxa_mortalidade_IM = 0.2
    if frames >= 719:
        nova_taxa_mort = 0
        #print('m = ' , nova_taxa_mort)
    else:
        Y = y_model_ref
        nova_taxa_mort = mu * Y[frames]
        #print('m = ' , nova_taxa_mort)

    return nova_taxa_mort





















"""
# Função Sazonal construída utilizando a Gaussiana que melhor se ajusta aos dados experimentais

df_gss = pd.read_excel(r'C:\Users\Vinicius\Documents\Vinícius\Faculdade\Iniciação Científica\Dados\Boxplots\Boxplot AC\Gaussiana\gaussIM.xlsx')

meses = df_gss['Meses'].values.tolist()
coleta = df_gss['Coleta'].values.tolist()

x = meses
y = coleta

#plt.scatter(x,y)

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
    

X = np.arange(0,730)

X_shift = []
for i in X:
    j = i + 50
    X_shift.append(j)

Z = []
for i in x_model:
    j = i + 0.5
    Z.append(j)




#plt.plot(X, y_model_ref, color = 'r')
#plt.plot(X_shift, y_model_ref, color = 'b')


#plt.plot(x_shift, odio_raiva_e_destruicao, color = 'r')
#plt.show()
plt.plot(x_model, y_model_norm, color = 'r')
plt.plot(Z, y_model_norm, color = 'b')
plt.show()
plt.gca().legend(('oviposição','mort. imaturos'))



    
def função_sazonal(frames):
    taxa = 1.0
    Y = y_model_norm        
    nova_taxa = taxa * Y[frames]

    print('nova taxa = ' , nova_taxa)    
    return nova_taxa

função_sazonal(180)






# Função Sazonal baseado num arco de seno

pi = math.pi

def função_sinoidal(x, phase):
    return abs(np.sin(x + phase))
    


def função_sazonal(frames):
    lista = []
    x = np.linspace(0, np.pi, 365)
    phase = 10.4
        
    y = função_sinoidal(x, phase)
    lista.append(y)
    
    plt.plot(x, y, lista)
    print(y[frames])
    

    return y[frames], lista


função_sazonal(280)

'''

    
    
    












