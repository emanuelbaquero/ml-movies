# %%
import pandas as pd
import numpy as np
import pandas_profiling 
import seaborn as sns
import matplotlib.pyplot as plt
from string import ascii_letters
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix,accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import unidecode as uni
import time
import pickle
pd.set_option('display.max_columns', None)

# %%
"""
## ARMAR DATASET PELICULAS
"""

# %%
data = pd.read_csv('data_movies_final.csv',sep='|').iloc[:,1:]
data['Rate'][data.Rate>10] = data.Rate[data.Rate>10]*0.10
data['fecha_mes'] = data['fecha'][data.fecha.notnull()]
dummies_meses = pd.get_dummies(data.fecha_mes[data.fecha_mes.notnull()].str.split(pat='-').apply(lambda x: x[1])).rename(columns={'01':'ENERO','02':'FEBRERO','03':'MARZO','04':'ABRIL','05':'MAYO','06':'JUNIO','07':'JULIO','08':'AGOSTO','09':'SEPTIEMBRE','10':'OCTUBRE','11':'NOVIEMBRE','12':'DICIEMBRE'})
data = pd.concat([data,dummies_meses],axis=1)
data = data[data.fecha_mes.notnull()]
data = data.drop(['IDDataLens','cant_baja_votos','cant_media_votos','cant_alta_votos','fecha_mes'],axis=1)
dataApropiado = pd.read_csv('data_apropiado.csv',sep='|').iloc[:,1:]
dataApropiado = dataApropiado.drop_duplicates()
data = data.merge(dataApropiado,how='left',left_on='Title',right_on='Title')
data['fecha_anio'] = data.fecha.str.split(pat='-').apply(lambda x: x[0])
data = data.drop(['popularity','largo_letras_titulo'],axis=1).drop_duplicates()
data = data[data.revenue!=0]
nuevaRate = pd.DataFrame(data.groupby(['Title'])['Rate'].mean()).reset_index()
data = data.drop(['Rate'],axis=1)
data = data.drop_duplicates()
data = data.merge(nuevaRate, how='left',left_on='Title',right_on='Title')
data = data.dropna()
data['fecha_anio'] = data.fecha_anio.astype(int)
data = data.drop(['fecha','Rate'],axis=1)

# %%
"""
## IMPORTAMOS LIBRERIAS SKLEARN
"""

# %%
# Regressions
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDClassifier, SGDRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier

# Boost Tuneados
from xgboost.sklearn import XGBRegressor,XGBClassifier
from catboost import CatBoostRegressor,CatBoostClassifier

# %%
"""
## TEST GRIDSEARCH SKLEARN
"""

# %%
def test_regresiones_sklearn(p_X, p_y):
    modelos_regresion = [LinearRegression(),
                         Ridge(),
                         Lasso(),
                         AdaBoostRegressor(),
                         GradientBoostingRegressor(),
                         RandomForestRegressor(),
                         XGBRegressor(),
                         CatBoostRegressor()]
    
    X = p_X
    y = p_y
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    
    parametros_LinealRegression = {}
    parametros_Ridge = {'alpha':[0.1,0.5,1]}
    parametros_Lasso = {'alpha':[1]}
    parametros_AdaBoostRegressor = {'random_state':[42],'n_estimators':[2,5]}
    parametros_GradientBoostRegressor = {'random_state':[54],'max_depth':[11],'n_estimators':[49]}
    parametros_RandomForestRegressor = {'random_state':[46],'max_depth':[44],'n_estimators':[44]}
    parametros_XGBoostRegressor = {'max_depth':[16],'iterations':[2],'alpha':[0],'learning_rate':[0.4]}
    parametros_CatboostRegressor = {'max_depth':[5],'iterations':[5],'learning_rate':[0.3]}
    
    dicccionario_modelos = {0:'Regresiones Lineal',1:'Modelo Ridge',2:'Modelo Lasso',3:'AdaBoostRegressor',4:'GradientBoostingRegressor',5:'RandomForestRegressor',6:'XGBRegressor',7:'CatBoostRegressor'}
    diccionario_parametros = {0:parametros_LinealRegression,1:parametros_Ridge,2:parametros_Lasso,3:parametros_AdaBoostRegressor,4:parametros_GradientBoostRegressor,5:parametros_RandomForestRegressor,6:parametros_XGBoostRegressor,7:parametros_CatboostRegressor}
    
    lista_indice = []
    lista_modelo = []
    lista_hiperparametros = []
    lista_r2 = []
    lista_tiempo = []
    lista_modelos = []
    
    
    for i, model in enumerate(modelos_regresion):
        start = pd.to_datetime(time.ctime())
        clf = GridSearchCV(model,diccionario_parametros[i])
        modelo = clf.fit(X_train, y_train)
        predicciones = modelo.predict(X_test)
        print('El R2 para '+dicccionario_modelos[i]+' es de: '+str(r2_score(y_test, predicciones)))
        fig, ax = plt.subplots()
        ax.scatter(y_test, predicciones)
        ax.set_xlabel('Measured')
        ax.set_ylabel('Predicted')
        plt.show()
        lista_indice.append(i)
        lista_modelo.append(dicccionario_modelos[i])
        lista_hiperparametros.append(clf.best_params_)
        lista_r2.append(r2_score(y_test, predicciones))
        print(r2_score(y_test, predicciones))
        lista_modelos.append(modelo)
        end = pd.to_datetime(time.ctime())
        lista_tiempo.append(str((end-start).total_seconds()) + ' segundos')
    
    return {'indices':lista_indice,'modelos':lista_modelo,'hiperparametros':lista_hiperparametros,'r2':lista_r2,'tiempo':lista_tiempo,'modelos_train':lista_modelos}

# %%
"""
## PREPARAMOS LAS X y
"""

# %%
data = data.dropna()
data = data.drop_duplicates()
X = data.drop(['Title','revenue'],axis=1)
y = data.revenue

# %%
"""
## RESULTADOS R2 REGRESIONES
"""

# %%
resultados = test_regresiones_sklearn(X,y)
resultados_r2 = pd.DataFrame({'indices':resultados['indices'],'nombres':resultados['modelos'],'hiperparametros':resultados['hiperparametros'],'r2':resultados['r2'],'tiempo':resultados['tiempo'],'modelos':resultados['modelos_train']})
resultados_r2

# %%
with open('resultados_regresiones_ganancias.pickle', 'wb') as handle:
    pickle.dump(resultados_r2, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %%
resultados_r2

# %%
"""
## ANALISIS DE RESULTADOS
"""

# %%
from xgboost.sklearn import XGBRegressor,XGBClassifier
XGBRegressor = XGBRegressor(alpha=0,max_depth=16,iterations=2,learning_rate=0.4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
modelo = XGBRegressor.fit(X_train,y_train)
predict = modelo.predict(X_test)
print('R2: '+str(r2_score(y_test,predict)))
import lime
import lime.lime_tabular
df_new = X_test

# %%
with open('resultados_regresiones_revenue.pickle', 'wb') as handle:
    pickle.dump(XGBRegressor, handle, protocol=pickle.HIGHEST_PROTOCOL)
df_array = df_new.sample(1)
XGBRegressor.predict(df_array)
import shap  
shap.initjs()

# %%
plt.figure(figsize=(40,35))
importances = XGBRegressor.feature_importances_
indices = np.argsort(importances)
features = X_train.columns
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

# %%
explainer = shap.TreeExplainer(XGBRegressor)
shap_values = explainer.shap_values(df_array)

# %%
df_array

# %%
features = X_train.columns
shap.force_plot(explainer.expected_value, shap_values, df_array,feature_names=features,out_names=['revenue_predicho'])

# %%
explainer = shap.TreeExplainer(XGBRegressor)
shap_values = explainer.shap_values(X_test.sample(1000))

# %%
shap.summary_plot(shap_values, X_test.sample(1000), plot_type="bar",color='black')

# %%
shap.summary_plot(shap_values, X_test.sample(1000))

# %%
shap.force_plot(explainer.expected_value, shap_values,X)

# %%


# %%


# %%


# %%
