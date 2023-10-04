# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 21:44:46 2023

@author: Vinicius
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import math
import xlwt
from xlwt import Workbook
from scipy.optimize import curve_fit

pi = math.pi

random.seed(456543)
np.random.seed(456543)


#21409 bom p imaturos
#4 muito boa
#5
#6 boa demais!
#4325

#Matriz Desenvolvimento 1314
dfIM = pd.read_excel(r'C:\Users\Vinicius\Documents\Vinícius\Faculdade\Iniciação Científica\Dados\Automata\Croquis\Remanejados\IM Roll 1.xlsx')
p1IM = dfIM['p1'].values.tolist()
p2IM = dfIM['p2'].values.tolist()
p3IM = dfIM['p3'].values.tolist()
p4IM = dfIM['p4'].values.tolist()
p5IM = dfIM['p5'].values.tolist()
p6IM = dfIM['p6'].values.tolist()
p7IM = dfIM['p7'].values.tolist()
p8IM = dfIM['p8'].values.tolist()
p9IM = dfIM['p9'].values.tolist()
p10IM = dfIM['p10'].values.tolist()
matIM = np.c_[p1IM, p2IM, p3IM, p4IM, p5IM, p6IM, p7IM, p8IM, p9IM, p10IM]

#Matriz Adultos 1314
dfAD = pd.read_excel(r'C:\Users\Vinicius\Documents\Vinícius\Faculdade\Iniciação Científica\Dados\Automata\Croquis\Remanejados\AD Roll 1.xlsx')    
p1AD = dfAD['p1'].values.tolist()
p2AD = dfAD['p2'].values.tolist()
p3AD = dfAD['p3'].values.tolist()
p4AD = dfAD['p4'].values.tolist()
p5AD = dfAD['p5'].values.tolist()
p6AD = dfAD['p6'].values.tolist()
p7AD = dfAD['p7'].values.tolist()
p8AD = dfAD['p8'].values.tolist()
p9AD = dfAD['p9'].values.tolist()
p10AD = dfAD['p10'].values.tolist()
matAD = np.c_[p1AD, p2AD, p3AD, p4AD, p5AD, p6AD, p7AD, p8AD, p9AD, p10AD]



#no desprop
def matriz_inicial(matrix):
    matriz_in = np.copy(matrix)
    
    return matriz_in


    


def DistProp(matrix,funcprop):
    B = np.zeros((100,100))
    desprop = funcprop(matrix)
    nova = np.copy(B)
    
    k0 = 0            
    k1 = 0
    k2 = 0
    k3 = 10
    k4 = 0
    k5 = 10
    while k1 <= 9:
        for i in range(0, desprop[k0][k1]):
            numI1 = random.randrange(k2, k3) 
            numJ1 = random.randrange(k4, k5)
            nova[numI1][numJ1] = 2
        k1 = k1 + 1
        k4 = k4 + 10
        k5 = k5 + 10
    k0 = k0 + 1
    k1 = 0
    k2 = k2 + 10
    k3 = k3 + 10
    k4 = 0
    k5 = 10
    while k1 <= 9:
        for i in range(0, desprop[k0][k1]):
            numI1 = random.randrange(k2, k3) 
            numJ1 = random.randrange(k4, k5)
            nova[numI1][numJ1] = 2
        k1 = k1 + 1
        k4 = k4 + 10
        k5 = k5 + 10
    k0 = k0 + 1
    k1 = 0
    k2 = k2 + 10
    k3 = k3 + 10
    k4 = 0
    k5 = 10
    while k1 <= 9:
        for i in range(0, desprop[k0][k1]):
            numI1 = random.randrange(k2, k3) 
            numJ1 = random.randrange(k4, k5)
            nova[numI1][numJ1] = 2
        k1 = k1 + 1
        k4 = k4 + 10
        k5 = k5 + 10
    k0 = k0 + 1
    k1 = 0
    k2 = k2 + 10
    k3 = k3 + 10
    k4 = 0
    k5 = 10
    while k1 <= 9:
        for i in range(0, desprop[k0][k1]):
            numI1 = random.randrange(k2, k3) 
            numJ1 = random.randrange(k4, k5)
            nova[numI1][numJ1] = 2
        k1 = k1 + 1
        k4 = k4 + 10
        k5 = k5 + 10
    k0 = k0 + 1
    k1 = 0
    k2 = k2 + 10
    k3 = k3 + 10
    k4 = 0
    k5 = 10
    while k1 <= 9:
        for i in range(0, desprop[k0][k1]):
            numI1 = random.randrange(k2, k3) 
            numJ1 = random.randrange(k4, k5)
            nova[numI1][numJ1] = 2
        k1 = k1 + 1
        k4 = k4 + 10
        k5 = k5 + 10        
    k0 = k0 + 1
    k1 = 0
    k2 = k2 + 10
    k3 = k3 + 10
    k4 = 0
    k5 = 10
    while k1 <= 9:
        for i in range(0, desprop[k0][k1]):
            numI1 = random.randrange(k2, k3) 
            numJ1 = random.randrange(k4, k5)
            nova[numI1][numJ1] = 2
        k1 = k1 + 1
        k4 = k4 + 10
        k5 = k5 + 10        
    k0 = k0 + 1
    k1 = 0
    k2 = k2 + 10
    k3 = k3 + 10
    k4 = 0
    k5 = 10
    while k1 <= 9:
        for i in range(0, desprop[k0][k1]):
            numI1 = random.randrange(k2, k3) 
            numJ1 = random.randrange(k4, k5)
            nova[numI1][numJ1] = 2
        k1 = k1 + 1
        k4 = k4 + 10
        k5 = k5 + 10        
    k0 = k0 + 1
    k1 = 0
    k2 = k2 + 10
    k3 = k3 + 10
    k4 = 0
    k5 = 10
    while k1 <= 9:
        for i in range(0, desprop[k0][k1]):
            numI1 = random.randrange(k2, k3) 
            numJ1 = random.randrange(k4, k5)
            nova[numI1][numJ1] = 2
        k1 = k1 + 1
        k4 = k4 + 10
        k5 = k5 + 10        
    k0 = k0 + 1
    k1 = 0
    k2 = k2 + 10
    k3 = k3 + 10
    k4 = 0
    k5 = 10
    while k1 <= 9:
        for i in range(0, desprop[k0][k1]):
            numI1 = random.randrange(k2, k3) 
            numJ1 = random.randrange(k4, k5)
            nova[numI1][numJ1] = 2
        k1 = k1 + 1
        k4 = k4 + 10
        k5 = k5 + 10
    k0 = k0 + 1
    k1 = 0
    k2 = k2 + 10
    k3 = k3 + 10
    k4 = 0
    k5 = 10
    while k1 <= 9:
        for i in range(0, desprop[k0][k1]):
            numI1 = random.randrange(k2, k3) 
            numJ1 = random.randrange(k4, k5)
            nova[numI1][numJ1] = 2
        k1 = k1 + 1
        k4 = k4 + 10
        k5 = k5 + 10
    
    return nova
    

IMprop = DistProp(matIM, matriz_inicial)
ADprop = DistProp(matAD, matriz_inicial)




def CI_adulto(matrix):
    CI_AD = np.copy(matrix)
    n = len(CI_AD)
    for i in range(0, n):
        for j in range(0, n):
            if CI_AD[i][j] == 2:
                distAD = random.randrange(0,100)
                if distAD <= 30:
                    CI_AD[i][j] = 1
    return CI_AD
    
    
CI_Adulto = CI_adulto(ADprop)


#print(CI_Adulto)
#print(IMprop)
    



def check_contorno(matrix, index_x, index_y, vizinho):
    if index_x + vizinho[0] < matrix.shape[0] and index_y + vizinho[1] < matrix.shape[1]:
        return True


def vizinhos_vazios_am(matrix, index_x, index_y):
    lista_true_vizinhos_ovp = []
    
    lista_de_duplas = [[-1,-1] , [-1,0] , [-1, +1], [0,-1] , [0, +1] , [+1, -1] , [+1, 0] , [+1,+1]]

    for item in lista_de_duplas:
        contorno_ = check_contorno(matrix, index_x, index_y, item)
        
        if contorno_:
            if matrix[index_x + item[0]][index_y + item[1]] == 0:
                lista_true_vizinhos_ovp.append(item)
    
    return lista_true_vizinhos_ovp


def vizinhos_vazios_ovp(matrix, index_x, index_y):
    lista_true_vizinhos_ovp = []
    
    lista_de_duplas = [[-1,-1] , [-1,0] , [-1, +1], [0,-1] , [0, +1] , [+1, -1] , [+1, 0] , [+1,+1]]

    for item in lista_de_duplas:
        contorno_ = check_contorno(matrix, index_x, index_y, item)
        
        if contorno_:
            if matrix[index_x + item[0]][index_y + item[1]] == 0:
                lista_true_vizinhos_ovp.append(item)
    
    return lista_true_vizinhos_ovp
    

def vizinhos_vazios(matrix, index_x, index_y):
    lista_true_vizinhos = []
    
    #lista_de_duplas = [[-3,-3] , [-3,0] , [-3, +3], [0,-3] , [0, +3] , [+3, -3] , [+3, 0] , [+3,+3]]
    lista_de_duplas = [[-2,-2] , [-2,-1], [-2,0], [-2,1], [-2,+2], [-1,2], [0,2], [1,2], [2,2], [2,1], [2,0], [2,-1], [2,-2] , [1,-2], [0,-2], [-1,-2], [-1,-1], [-1,0], [-1,1], [0,1], [1,1], [1,0], [1,-1] , [0,-1]]

    for item in lista_de_duplas:
        contorno_ = check_contorno(matrix, index_x, index_y, item)
        
        if contorno_:
            if matrix[index_x + item[0]][index_y + item[1]] == 0:
                lista_true_vizinhos.append(item)
    
    return lista_true_vizinhos



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


ovp_shift = norm_crr_2[155:]
for i in range(0, 1):
    ovp_shift.append(0)
for i in norm_crr_2:
    ovp_shift.append(i)
del ovp_shift[365:]
    
mu_shift = []
for i in range(0,80):
    mu_shift.append(0)
for i in norm_crr_2:
    mu_shift.append(i)
del mu_shift[365:]


#X = np.arange(0,365)
#plt.plot(X, ovp_shift, color = 'r', label = 'oviposição')
#plt.plot(X, mu_shift, color = 'b', label = 'mortalidade imaturos')
#plt.gca().legend(('oviposição','mort. imaturos'))
#plt.title('Curvas Sobrepostas')
#plt.show()


def função_sazonal_ovp(frames):
    taxa = 0.4
    Y = ovp_shift
    nova_taxa_ovp = taxa * Y[frames]
    
    return nova_taxa_ovp


def função_sazonal_mort(frames):
    taxa = 0.1
    Y = mu_shift
    nova_taxa_mort = taxa * Y[frames]
    
    return nova_taxa_mort




def matriz_borda(matrix1, matrix2, frames):
    N = 102
    M = 102
    borda = 2
    bordaL = N - borda
    bordaC = M - borda
    
    matbordaIM = np.zeros((N, M))
    matbordaAD = np.zeros((N, M))
    for i in range(borda,bordaL):
        for j in range(borda,bordaC):
            matbordaIM[i][j] = matrix1[int(i)][int(j)]
            matbordaAD[i][j] = matrix2[int(i)][int(j)]
            
            northIM = matbordaIM[i][j-1] if j > 0 else 0
            southIM = matbordaIM[i][j+1] if j < (np.size(matbordaIM,1)-2) else 0
            westIM = matbordaIM[i+1][j] if i < (np.size(matbordaIM,0)-2) else 0
            eastIM = matbordaIM[i-1][j] if i > 0 else 0
            seIM = matbordaIM[i+1][j+1] if i < (np.size(matbordaIM,0)-2) and j < (np.size(matbordaIM,1)-2) else 0
            swIM = matbordaIM[i+1][j-1] if i < (np.size(matbordaIM,0)-2) and j > 0 else 0
            neIM = matbordaIM[i-1][j+1] if i > 0 and j < (np.size(matbordaIM,1)-2) else 0
            nwIM = matbordaIM[i-1][j-1] if i > 0  and j > 0 else 0
            neighboursIM = np.sum([northIM, southIM, westIM, eastIM, seIM, swIM, neIM, nwIM])
            
            cellAD = matbordaAD[i][j] #if j > 0 else 0
            northAD = matbordaAD[i][j-1] #if j > 0 else 0
            southAD = matbordaAD[i][j+1] #if j < (np.size(matbordaAD,1)-2) else 0
            westAD = matbordaAD[i+1][j] #if i < (np.size(matbordaAD,0)-2) else 0
            eastAD = matbordaAD[i-1][j] #if i > 0 else 0
            seAD = matbordaAD[i+1][j+1] #if i < (np.size(matbordaAD,0)-2) and j < (np.size(matbordaAD,1)-2) else 0
            swAD = matbordaAD[i+1][j-1] #if i < (np.size(matbordaAD,0)-2) and j > 0 else 0
            neAD = matbordaAD[i-1][j+1] #if i > 0 and j < (np.size(matbordaAD,1)-2) else 0
            nwAD = matbordaAD[i-1][j-1] #if i > 0  and j > 0 else 0
            
            R2_1 = matbordaAD[i-2][j-2]
            R2_2 = matbordaAD[i-2][j-1]
            R2_3 = matbordaAD[i-2][j]
            R2_4 = matbordaAD[i-2][j+1]
            R2_5 = matbordaAD[i-2][j+2]
            R2_6 = matbordaAD[i-1][j+2]
            R2_7 = matbordaAD[i][j+2]
            R2_8 = matbordaAD[i+1][j+2]
            R2_9 = matbordaAD[i+2][j+2]
            R2_10 = matbordaAD[i+2][j+1]
            R2_11 = matbordaAD[i+2][j]
            R2_12 = matbordaAD[i+2][j-1]
            R2_13 = matbordaAD[i+2][j-2]
            R2_14 = matbordaAD[i+1][j-2]
            R2_15 = matbordaAD[i][j-2]
            R2_16 = matbordaAD[i-1][j-2]
            
            neighboursAD = np.sum([cellAD, northAD, southAD, westAD, eastAD, seAD, swAD, neAD, nwAD, R2_1, R2_2, R2_3, R2_4, R2_5, R2_6, R2_7, R2_8, R2_9, R2_10, R2_11, R2_12, R2_13, R2_14, R2_15, R2_16])
            
            
            
            # oviposição dos adultos:

            if neighboursAD >= 4:
                index_x = i
                index_y = j
                vizinhos_possiveis_ovp = vizinhos_vazios_ovp(matrix1, index_x, index_y)
                num_vizinhos_ovp = len(vizinhos_possiveis_ovp)
                
                if num_vizinhos_ovp > 0:
                    numero_aleatorio_ovp = random.randrange(0, num_vizinhos_ovp) #np.random.choice(np.arange(num_vizinhos_ovp))
                    vizinho_escolhido_ovp = vizinhos_possiveis_ovp[numero_aleatorio_ovp]
                    ovp_sim = função_sazonal_ovp(frames)
                    ovp_nao = 1 - ovp_sim
                    #print('oviposição =' , ovp_sim)
                    opcoes = ['SIM' , 'NAO']
                    
                    resposta = np.random.choice(opcoes , p = [ovp_sim , ovp_nao])
                    
                    if resposta == 'SIM': 
                        matbordaIM[index_x + vizinho_escolhido_ovp[0]][index_y + vizinho_escolhido_ovp[1]] = 2
                        neighboursAD = 0                        
                            
            # dispersão dos adultos: 10m por mês em uma área de 100m², o que equivale a 3 subparcelas por dia
               # primeiro modelo : 1 subparcela por dia, Moore com r = 1
               # segundo modelo : 3 subparcelas por dia, Moore com r = 3


#            if matbordaAD[i][j] == 2:
 #               vizinhos_possiveis = vizinhos_vazios(matrix2, i , j)
  #              num_vizinhos = len(vizinhos_possiveis)
   #                 
    #            if num_vizinhos > 0:
     #               numero_aleatorio = np.random.choice(np.arange(num_vizinhos))
      #              vizinho_escolhido = vizinhos_possiveis[numero_aleatorio]
       #             #print('vizinho' , vizinho_escolhido)
        #            disp = random.uniform(0 , 1)
                    #print(disp)
         #           if disp <= 0.1:
                        #print('foi')
          #              matbordaAD[i][j] = 0
           #             matbordaAD[i + vizinho_escolhido[0]][j + vizinho_escolhido[1]] = 2
            
            
            
            

            # mortalidade dos imaturos: 0,03373      min: 0.0228    max: 0.0633   
            if matbordaIM[i][j] == 2:
                mort_sim = função_sazonal_mort(frames)                 # mortIM = random.uniform(0 , 1)
                mort_nao = 1 - mort_sim
                opcoes = ['SIM' , 'NAO']
                resposta = np.random.choice(opcoes , p = [mort_sim , mort_nao])
                if resposta == 'SIM': 
                    matbordaIM[i][j] = 0
                    
                    
            # emergência dos imaturos: 0,01834      
            if matbordaIM[i][j] == 2:
                vizinhos_possiveis_am = vizinhos_vazios_am(matrix2, i , j)
                num_vizinhos_am = len(vizinhos_possiveis_am)                
                if num_vizinhos_am > 0:
                    numero_aleatorio = np.random.choice(np.arange(num_vizinhos_am))
                    vizinho_escolhido_am = vizinhos_possiveis_am[numero_aleatorio]
                    emIM = random.uniform(0 , 1)
                    if emIM < 0.0124:
                        #print('amadurece' , emIM)
                        matbordaIM[i][j] = 0
                        matbordaAD[i + vizinho_escolhido_am[0]][j + vizinho_escolhido_am[1]] = 1
                #else:
                  #  print('nao amadurece' , emIM)


            # mortalidade dos adultos: 0,00840      min: 0.0034      max: 0.0166 
            if matbordaAD[i][j] == 2 or matbordaAD[i][j] == 1:
                mortAD = random.uniform(0 , 1)
                #print('mortad' , mortAD)
                if mortAD < 0.0034:
                    matbordaAD[i][j] = 0

            
            # amadurecimento dos adultos: 0,03571     min: 0.0238    max: 0.0714 
            if matbordaAD[i][j] == 1:
                amAD = random.uniform(0 , 1)
                #print('amAD' , amAD)
                if amAD < 0.0357:
                    matbordaAD[i][j] = 2
            # talvez o amadurecimentos dos adultos não funcione bem com parâmetros estocásticos...
            # uma ideia seria colocar um contador para cada adulto infértil.                    
                    
                    

                       
                        

    matbordaIM[0] = 0
    matbordaIM[1] = 0
    matbordaIM[-1] = 0
    matbordaIM[-2] = 0        
    matbordaIM[:,0] = 0
    matbordaIM[:,1] = 0
    matbordaIM[:,-1] = 0
    matbordaIM[:,-2] = 0
    
    matbordaAD[0] = 0
    matbordaAD[1] = 0
    matbordaAD[-1] = 0
    matbordaAD[-2] = 0        
    matbordaAD[:,0] = 0
    matbordaAD[:,1] = 0
    matbordaAD[:,-1] = 0
    matbordaAD[:,-2] = 0
    #print(check)        
            
    
    return matbordaIM, matbordaAD


IM, AD = matriz_borda(IMprop , CI_Adulto, 0)




def anima(frames, matrix1, matrix2):
    #fig = plt.figure(figsize=(10,10))

    for i in np.arange(frames):
        print('i' , i)        
        matrix1, matrix2 = matriz_borda(matrix1, matrix2, i)

    
    #Imaturos = fig.add_subplot(121)
    #plt.imshow(matrix1 , vmin = 0, vmax = 2, cmap = 'Greys')
    #Adultos = fig.add_subplot(122)
    #plt.imshow(matrix2 , vmin = 0, vmax = 2, cmap = 'Greys')

    #Imaturos.set_title('Imaturos')
    #Adultos.set_title('Adultos')
    #plt.axis('on')
    #plt.show()
    
    return matrix1, matrix2


#anima(20, IM, AD)


T = 364


wb_IM = Workbook()

sheet1 = wb_IM.add_sheet('Sheet 1')
sheet1.write(0, 0, 'Parcelas')

for i in range(1,101):
    sheet1.write(i, 0 , i)
    
datas = 0
for i in range(1,27):
    sheet1.write(0, i , datas)    
    datas = datas + 14

def Contador_IM(tempo, matrix1, matrix2):
    
    jIM = 1
    passo = 1
    

    for d in range(0, tempo, 14):    
        IMf, ADf = anima(passo, matrix1, matrix2)        
        contaIM = 0
        quantIM = []    
        
        i1 = 2
        i2 = 12
        j1 = 2
        j2 = 12
        
        iIM = 1
        
    
        while j1 < 102:
            while i1 < 102:
                for i in range(i1,i2):
                    for j in range(j1,j2):
                        if IMf[i][j] == 2:
                            contaIM = contaIM + 1
                quantIM.append(contaIM)
                                                                        
                i1 = i1 + 10
                i2 = i2 + 10
                contaIM = 0
                        
            i1 = 2
            i2 = 12
            j1 = j1 + 10
            j2 = j2 + 10
            
        print(quantIM)
        while iIM <= 100:
            for k in quantIM:
                sheet1.write(iIM , jIM ,k)
                iIM = iIM + 1
    
        jIM = jIM + 1
        passo = passo + 14
            


Contador_IM(T, IM, AD)    
wb_IM.save('Quantidades_Imaturos_Sphenophorus.xls')



wb_AD = Workbook()

sheet1 = wb_AD.add_sheet('Sheet 1')
sheet1.write(0, 0, 'Parcelas')

for i in range(1,101):
    sheet1.write(i, 0 , i)
    
datas = 0
for i in range(1,27):
    sheet1.write(0, i , datas)    
    datas = datas + 14

def Contador_AD(tempo, matrix1, matrix2):
    
    jAD = 1
    passo = 1
    
    for d in range(0, tempo, 14):    
        IMf, ADf = anima(tempo, matrix1, matrix2)
        contaAD = 0
        quantAD = []    
        
        i1 = 2
        i2 = 12
        j1 = 2
        j2 = 12
        
        iAD = 1
        
    
        while j1 < 102:
            while i1 < 102:
                for i in range(i1,i2):
                    for j in range(j1,j2):
                        if ADf[i][j] == 2 or ADf[i][j] == 1:
                            contaAD = contaAD + 1
                quantAD.append(contaAD)
                                                                        
                i1 = i1 + 10
                i2 = i2 + 10
                contaAD = 0
                        
            i1 = 2
            i2 = 12
            j1 = j1 + 10
            j2 = j2 + 10
            
        print(quantAD)
        while iAD <= 100:
            for k in quantAD:
                sheet1.write(iAD , jAD ,k)
                iAD = iAD + 1
    
        jAD = jAD + 1
        passo = passo + 14
            


Contador_AD(T, IM, AD)    
wb_AD.save('Quantidades_Adultos_Sphenophorus.xls')



import time
start_time = time.time()
print("--- %s seconds ---" % (time.time() - start_time))
















