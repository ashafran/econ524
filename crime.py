# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 13:49:15 2020

@author: aric
"""


import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

d=pd.read_csv("london_crime.csv")
d["crimerate"]=d.crime/d.population
d["policerate"]=d.police/d.population
d["lcrime"]=np.log(d.crimerate)
d["lpolice"]=np.log(d.policerate)
d["lemp"]=np.log(d.emp)
d["lun"]=np.log(d.un)
d["lymale"]=np.log(d.ymale)
d["lwhite"]=np.log(d.white)


model1=smf.ols("lcrime ~ lpolice + lemp + lun + lymale +lwhite",data=d).fit()

e=pd.DataFrame(d.values[d.week>52,11:17]-d.values[d.week<53,11:17])
e.columns=["dlcrime","dlpolice","dlemp","dlun","dlymale","dlwhite"]
e["week"]=np.array(d.week[d.week>52])
model2=smf.ols("dlcrime ~ dlpolice + dlemp + dlun + dlymale +dlwhite+C(week)",data=e).fit()


e["sixweeks"]=np.array((d[d.week>52].week>79) & (d[d.week>52].week<86),dtype=int)
e["treat"]=np.array((d[d.week>52].borough==1) | (d[d.week>52].borough==2) | (d[d.week>52].borough==3)|(d[d.week>52].borough==6)|(d[d.week>52].borough==14),dtype=int)
e["sixweeks_treat"]=e.sixweeks*e.treat
#e["post"]=np.array((d[d.week>52].week>85),dtype=int)
#e["post_treat"]=e.post*e.treat
model3=smf.ols("dlcrime ~ sixweeks + sixweeks_treat + C(week) + dlemp + dlun + dlymale +dlwhite",data=e).fit()

fulldata=d
diffdata=e
