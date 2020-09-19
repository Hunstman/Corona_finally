import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import numba as nb


######################
# ITALY TREE
######################

Italia = {
  "Lombardia" : {
    "index" : 0,
    "edges" : [2,4,7,1,19],
    "pop" : 10060,  
  },
  "Veneto" : {
    "index" : 1,
    "edges" : [0,7,6,2],
    "pop" : 4905,

  },
  "Emilia Romagna" : {
    "index" : 2,
    "edges" : [0,1,5,4,10,8],
          "pop" : 4459,

  },
  "Aosta" : {
    "index" : 3,
    "edges" : [4],
          "pop" : 125,

  },
  "Piemonte" : {
    "index" : 4,
    "edges" : [0,3,5,2],
          "pop" : 4356,

  },
  "Liguria" : {
    "index" : 5,
    "edges" : [4,2,8,19],
          "pop" : 1550,

  },
  "Friuli Venezia Giulia" : {
    "index" : 6,
    "edges" : [1],
          "pop" : 1215,

  },
  "Trentino" : {
    "index" : 7,
    "edges" : [1,0],
          "pop" : 1072,

  },
  "Toscana" : {
    "index" : 8,
    "edges" : [9,5,2,10,11],
          "pop" : 3729,

  },
  "Umbria" : {
    "index" : 9,
    "edges" : [8,11,10],
          "pop" : 882,

  },
  "Marche" : {
    "index" : 10,
    "edges" : [8,2,11,15,9],
          "pop" : 1525,

  },
  "Lazio" : {
    "index" : 11,
    "edges" : [8,9,10,14,12,15,19],
          "pop" : 5879,

  },
  "Molise" : {
    "index" : 12,
    "edges" : [15,14,17,11],
          "pop" : 305,

  },
  "Basilicata" : {
    "index" : 13,
    "edges" : [14,17,16],
          "pop" : 562,

  },
  "Campania" : {
    "index" : 14,
    "edges" : [11,13,12,17],
          "pop" : 5800,

  },
  "Abruzzo" : {
    "index" : 15,
    "edges" : [11,10,12],
          "pop" : 1311,

  },
  "Calabria" : {
    "index" : 16,
    "edges" : [14,13,18],
          "pop" : 1947,

  },
  "Puglia" : {
    "index" : 17,
    "edges" : [14,12,13],
          "pop" : 4029,

  },
  "Sicilia" : {
    "index" : 18,
    "edges" : [16],
          "pop" : 5000,

  },
  "Sardegna" : {
    "index" : 19,
    "edges" : [5,0,11],
          "pop" : 1639,

  }
}



#########################
#parametri
##########################

mu_i = 7e-3#mortalità dei sintomatici
beta_i = 4*mu_i*1e-3#infettività degli infetti sintomatici 4/1000 morte
beta_a = 8*mu_i*1e-3#infettività degli infetti asintomatici 4/1000 morte
sigma_a = 2*beta_i*0.1#diffusione asintomatici tra regioni
sigma_i = beta_i*0.1#diffusione sintomatici tra regioni
mu_i = 7e-6#mortalità dei sintomatici
mu_a = 0#mortalità degli asintomatici
g_i = 4/3*mu_i#rate guarigione degli infetti 4/3 morte
g_a = 4/3*mu_i#rate guarigione degli asintomatici 4/3 morte
v_i = 10*mu_i#tasso latenti -> sintomatici 10 morte
v_a = 2*10*mu_i#tasso latenti -> asintomatici 10 morte 







######################################
#Time Evolution
######################################

def evoluzione_temporale(T,Italia,beta_a,beta_i,sigma_a,sigma_i,mu_a,mu_i,g_i,g_a,v_i,v_a):
    #Matrice con tutto
    dt = 0.01
    timestep = int(T/dt)
    total = np.zeros((timestep,20,10))
    
    #Inizializzazione popolazione totale per regione
    for regione in Italia:
        total[0,Italia[regione]['index'],0]=Italia[regione]['pop']
        total[-1,Italia[regione]['index'],-2] = Italia[regione]['index']
        total[-1,Italia[regione]['index'],-1] = timestep
    #Casi in Lombardia t=0
    total[0,Italia['Lombardia']['index'],2]=1./1e3#latente
    
    for i in range(timestep-1):    
        for regione in Italia: 
            total[i,Italia[regione]['index'],-2] = Italia[regione]['index']
            total[i,Italia[regione]['index'],-1] = i

           # total[i, Italia[regione][indices],np.where(total[i, Italia[regione][indices]]< 0)] = 0
            asinto_reg_vicina = 0
            sinto_reg_vicina = 0
            for indices in Italia[regione]['edges']:
                asinto_reg_vicina += total[i,indices,2]*total[i, Italia[regione]['index'],0]
                sinto_reg_vicina += total[i,indices,3]*total[i, Italia[regione]['index'],0]

        #Equazioni differenziali needed
            #Sani
            total[i+1, Italia[regione]['index'],0] = total[i, Italia[regione]['index'],0] -\
               dt*beta_i*total[i, Italia[regione]['index'],0]*total[i, Italia[regione]['index'],3] - \
                dt*beta_a*total[i, Italia[regione]['index'],0]*total[i, Italia[regione]['index'],2] \
                -dt*sigma_a*asinto_reg_vicina -dt*sigma_i*sinto_reg_vicina

            #Latenti
            total[i+1, Italia[regione]['index'],1] = total[i, Italia[regione]['index'],1] - (total[i+1, Italia[regione]['index'],0] - total[i, Italia[regione]['index'],0]) \
            -v_i*dt*total[i, Italia[regione]['index'],1] - v_a*dt*total[i, Italia[regione]['index'],1]

            #Asintomatici
            total[i+1, Italia[regione]['index'],2] = total[i, Italia[regione]['index'],2] +\
                dt*total[i, Italia[regione]['index'],1]*v_a - dt*(mu_a+g_a)*total[i, Italia[regione]['index'],2]

            #Sintomatici
            total[i+1, Italia[regione]['index'],3] = total[i, Italia[regione]['index'],3] +\
                dt*total[i, Italia[regione]['index'],1]*v_i - dt*(mu_i+g_i)*total[i, Italia[regione]['index'],3]

            #Guariti
            total[i+1, Italia[regione]['index'],4] = total[i, Italia[regione]['index'],4] +\
                dt*total[i, Italia[regione]['index'],3]*g_i + dt*g_a*total[i, Italia[regione]['index'],2]
            
            #Morti
            total[i+1, Italia[regione]['index'],5] =  total[i, Italia[regione]['index'],5] +\
                dt*mu_i*total[i, Italia[regione]['index'],3] +dt*mu_a*total[i, Italia[regione]['index'],2]
            
            #Totale infetti 
            total[i+1, Italia[regione]['index'],6] =  total[i, Italia[regione]['index'],6] + total[i, Italia[regione]['index'],2] +\
                total[i, Italia[regione]['index'],3]
            
    return total

#########################
#Executing
#########################

T=5000
total_final = evoluzione_temporale(T,Italia,beta_a,beta_i,sigma_a,sigma_i,mu_a,mu_i,g_i,g_a,v_i,v_a)



#########################
#To Plot
#########################

title = ["Sani", "Latenti", "Asintomatici", "Sintomatici", "Guariti", "Morti"]

for i in range(6):
    plt.title(title[i])
    plt.plot(total_final[:,0,i])
    plt.show()
plt.title("Totale infetti Lombardia")
plt.plot(total_final[:,0,2] + total_final[:,0,3])
plt.show()
#Per plottare i morti
plt.plot(total_final[:,0,2])
plt.show()

##########################
#To Save Data
##########################

tabella = np.ndarray.tolist(total_final.reshape((total_final.shape[0]*total_final.shape[1],-1)))
for regione in Italia:
        for i in range(len(tabella)):
            if tabella[i][8] == Italia[regione]['index']:
                tabella[i][8]=regione

##########################
#To export Data
##########################

tabella_to_export =[]
tabella_to_export.append(["Sani", "Latenti", "Asintomatici", "Sintomatici", "Guariti", "Morti","CASI TOTALI", "regione","time"])
value=0
for i in range(0,len(tabella),200000):
    for j in range(i,i+20):
        tabella_to_export.append(tabella[j])

import csv

with open('dati_modello.csv', 'wb') as csvfile:
    w = csv.writer(csvfile, delimiter=',')
    for row in tabella_to_export:
        w.writerow(row)
