# Paquetes necesarios para el analisis de datos 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import janitor as jn 
from pathlib import Path

# Librerias de ML
from sklearn.model_selection import train_test_split     # Entrenamiento y prueba
from sklearn.preprocessing import StandardScaler         # Estandarizar variables númericas
from sklearn.compose import ColumnTransformer            # Escalar variables numericas
from sklearn.pipeline import Pipeline
# LASSO
from sklearn.linear_model import Lasso, LassoCV          # Permite calcular el valor de alfa mediante cross-validation
from sklearn.impute import SimpleImputer
# Metricas 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  
# Random Forest 
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



#-------------------------- CARGA E INICIO DEL PROCESO ------------------------------#

# Fijamos la ruta de trabajo para el dataset y los gráficos del informe 
ruta_base = Path('C:/Users/gaboa/OneDrive/01. Estudios/03.Magister/02. Magister_Estadistica/II Semestre/Machine Learning/Veloz/Trabajo_Final')
ruta_graficos = ruta_base/'figures'
# Generamos la carpeta en caso de no existir 
mkdirs = [ruta_graficos]
for carpeta in mkdirs:
    if not carpeta.exists():
        carpeta.mkdir(parents=True)
        print(f'Se crea carpeta: {carpeta}')
    else:
        print(f'La carpeta {carpeta} ya existe')

# Importamos la base de datos 
df = pd.read_csv(ruta_base/'epl_final.csv').clean_names()




#-------------------------- DESCRIPCION DEL DATASET -------------------------------#

# Vemos el tamaño y la informacion general (detectando nro de datos nulos)
print(f'Las dimensiones de la base de datos es {df.shape}')

# Analizamos los tipos de variables que contiene el datset 
print(df.dtypes)

# Revisamos el nombre de las variables que contiene el dataset
print(df.columns)
#Dejamos guardado las variables originales de la base 
var_pre_edit = df.columns

# Contabilizamos el número de datos perdidos del datset 
print(f'El número de datos nan por variable: {df.isna().sum()}')

# Contabilizamos observaciones duplicadas 
num_dupli = df.duplicated().sum()
print(f"Número de filas duplicadas completas: {num_dupli}")

# Convertimos la variable matchdate a formato fecha 
df['matchdate'] = pd.to_datetime(df['matchdate'], errors='raise', dayfirst=True)
#Corroboramos el cambio
print(f"El formado de la variable matchdate es: {df['matchdate'].dtype}")

# revisamos como se ve el dataset
df.head()




#---------------- CREACION DE VARIABLES PARA EL MODELO -----------------#

# Generamos las temporadas según calendario europeo
df['season'] = np.where(df['matchdate'].dt.month >= 8,
                        df['matchdate'].dt.year,
                        df['matchdate'].dt.year - 1)

# Primero ordenamos 
df = df.sort_values(['season', 'matchdate'])

# Generamos la asignacion de puntos por partido ganado o empatado
df['home_points'] = np.select(
    [df['fulltimehomegoals'] > df['fulltimeawaygoals'],   # gana local
     df['fulltimehomegoals'] == df['fulltimeawaygoals']], # empate
    [3, 1],
    default=0)

# Puntos del equipo visita en ese partido
df['away_points'] = np.select(
    [df['fulltimeawaygoals'] > df['fulltimehomegoals'],   # gana visita
     df['fulltimeawaygoals'] == df['fulltimehomegoals']], # empate
    [3, 1],
    default=0)

# Forma o condición deportiva reciente (puntos últimos 5 partidos dentro de la temporada)
df['home_form_last5'] = (df.groupby(['hometeam'])['home_points'].transform(lambda x: x.shift().rolling(5, min_periods=1).mean()))
df['away_form_last5'] = (df.groupby(['awayteam'])['away_points'].transform(lambda x: x.shift().rolling(5, min_periods=1).mean()))

# Rendimiento ofensivo (goles a favor últimos 5 partidos)
df['home_attack_last5'] = (df.groupby(['hometeam'])['fulltimehomegoals'].transform(lambda x: x.shift().rolling(5, min_periods=1).mean()))
df['away_attack_last5'] = (df.groupby(['awayteam'])['fulltimeawaygoals'].transform(lambda x: x.shift().rolling(5, min_periods=1).mean()))

# Rendimiento defensivo (goles en contra últimos 5 partidos)
df['home_defense_last5'] = (df.groupby(['hometeam'])['fulltimeawaygoals'].transform(lambda x: x.shift().rolling(5, min_periods=1).mean()))
df['away_defense_last5'] = (df.groupby(['awayteam'])['fulltimehomegoals'].transform(lambda x: x.shift().rolling(5, min_periods=1).mean()))

# Revisamos valores nulos de nuestras nuevas variables (deberian existir según la lógica de las variables rendimiento)
print(df[['home_form_last5', 'away_form_last5','home_attack_last5', 
          'away_attack_last5', 'home_defense_last5', 'away_defense_last5']].isna().sum())

# Precisión de tiros (evitando división por cero)
df['shot_acc_home'] = np.where(df['homeshots'] > 0,
                               df['homeshotsontarget'] / df['homeshots'],
                               0)

df['shot_acc_away'] = np.where(df['awayshots'] > 0,
                               df['awayshotsontarget'] / df['awayshots'],
                               0)

# Generamos dos dataframe para calcular los puntos
home_pts = df[['season', 'matchdate', 'hometeam', 'home_points']].rename(
    columns={'hometeam': 'team', 'home_points': 'points'})

away_pts = df[['season', 'matchdate', 'awayteam', 'away_points']].rename(
    columns={'awayteam': 'team', 'away_points': 'points'})
# Juntamos ambos dataframes
points_long = pd.concat([home_pts, away_pts], ignore_index=True)

# Ordenamos por equipo y fecha
points_long = points_long.sort_values(['season', 'team', 'matchdate'])

# Cumulative sum y luego shift(1) para que sean "puntos antes de este partido"
points_long['cum_points_before'] = (points_long.groupby(['season', 'team'])['points'].cumsum().shift(1))
# El primer partido de cada equipo tendrá NaN, lo reemplazamos por 0
points_long['cum_points_before'] = points_long['cum_points_before'].fillna(0)

# Merge de puntos acumulados al df original
df = df.merge(
    points_long[['season', 'team', 'matchdate', 'cum_points_before']],
    how='left',
    left_on=['season', 'hometeam', 'matchdate'],
    right_on=['season', 'team', 'matchdate']
).rename(columns={'cum_points_before': 'home_points_before'})

df = df.merge(
    points_long[['season', 'team', 'matchdate', 'cum_points_before']],
    how='left',
    left_on=['season', 'awayteam', 'matchdate'],
    right_on=['season', 'team', 'matchdate']
).rename(columns={'cum_points_before': 'away_points_before'})

# Nos aseguramos de que los partidos estén ordenados en el tiempo
df = df.sort_values('matchdate').reset_index(drop=True)

# Función auxiliar para crear eficacia histórica por grupo (equipo)
def add_rolling_effectiveness(df, team_col, goals_col, shots_on_target_col,
                              prefix, windows=(3, 10)):
    # Agrupamos por equipo
    grouped = df.groupby(team_col)

    for w in windows:
        # Suma de goles y tiros al arco en los últimos w partidos ANTERIORES
        goals_rolling = grouped[goals_col].apply(lambda s: s.shift().rolling(window=w, min_periods=1).sum())
        shots_rolling = grouped[shots_on_target_col].apply(lambda s: s.shift().rolling(window=w, min_periods=1).sum())
        # Eficacia = goles acumulados / tiros al arco acumulados
        eff = goals_rolling / shots_rolling
        new_col = f'{prefix}_last{w}'
        df[new_col] = eff.reset_index(level=0, drop=True)
    return df

# Eficacia histórica del equipo LOCAL (solo partidos jugando de local)
df = add_rolling_effectiveness(
    df,
    team_col='hometeam',
    goals_col='fulltimehomegoals',
    shots_on_target_col='homeshotsontarget',
    prefix='eff_on_target_home',
    windows=(3, 10))

# Eficacia histórica del equipo VISITANTE (solo partidos jugando de visita)
df = add_rolling_effectiveness(
    df,
    team_col='awayteam',
    goals_col='fulltimeawaygoals',
    shots_on_target_col='awayshotsontarget',
    prefix='eff_on_target_away',
    windows=(3, 10))



# ================== CÁLCULO DE RATING ELO ================== #
# Nos aseguramos del orden temporal por temporada
df = df.sort_values(['season', 'matchdate']).reset_index(drop=True)
# Parámetros del sistema Elo
# rating inicial
ELO_INIT = 1500      
# sensibilidad del sistema /decisión arbitaria
K = 20 
# ventaja de local en puntos Elo              
HOME_ADV = 100       
# Diccionario para ir guardando el rating de cada equipo
elo_ratings = {}     
# Listas donde iremos almacenando el Elo ANTES del partido
elo_home_before = []
elo_away_before = []

for row in df.itertuples(index=False):
    home = row.hometeam
    away = row.awayteam
    gh = row.fulltimehomegoals
    ga = row.fulltimeawaygoals
    # Si el equipo no tiene rating aún, parte en ELO_INIT
    r_home = elo_ratings.get(home, ELO_INIT)
    r_away = elo_ratings.get(away, ELO_INIT)
    # Guardamos los ratings PRE-partido como variables del modelo
    elo_home_before.append(r_home)
    elo_away_before.append(r_away)
    # Resultado real del partido (puntuación Elo)
    if gh > ga:
        s_home, s_away = 1.0, 0.0     
    elif gh == ga:
        s_home, s_away = 0.5, 0.5     
    else:
        s_home, s_away = 0.0, 1.0     
    # Resultado esperado según Elo (incluimos ventaja de local)
    # E_home = 1 / (1 + 10^((R_away - (R_home + HOME_ADV))/400))
    expected_home = 1 / (1 + 10 ** (((r_away) - (r_home + HOME_ADV)) / 400))
    expected_away = 1 - expected_home
    # Actualizamos ratings
    new_r_home = r_home + K * (s_home - expected_home)
    new_r_away = r_away + K * (s_away - expected_away)
    elo_ratings[home] = new_r_home
    elo_ratings[away] = new_r_away
    
# Añadimos las columnas al dataframe
df['elo_home_before'] = elo_home_before
df['elo_away_before'] = elo_away_before
df['elo_diff_before'] = df['elo_home_before'] - df['elo_away_before']


# Revisamos como quedo para realizar tabla resumen 
len(df.columns)
df.dtypes.value_counts()



#---------------------------------- CREACION DE VARIABLES TARGET ----------------------------------#

# Generamos nuestra variable target -- diferencial de goles -- Regresion
df['goal_diff'] = df['fulltimehomegoals'] - df['fulltimeawaygoals']
# Generamos el target para Random Forest
df['result_class'] = df['goal_diff'].apply(lambda x: 0 if x > 0 else (1 if x == 0 else 2))
# Calculamos la media de nuestra variable target 
mean_goal_diff = df['goal_diff'].mean()
print(mean_goal_diff)

# Será binario: gana local (1) o no (0) - Random Forest
df['home_win_bin'] = (df['goal_diff'] > 0).astype(int)
# Contabilizamos el numero de resultados de nuestra data 
count_win_local = df['home_win_bin'].value_counts()
print(count_win_local)

# Vemos solo las variables nuevas 
var_post_edit = df.columns
vars_new = list(set(var_post_edit) - set(var_pre_edit))
df_new_vars = df[vars_new]




#---------------------------  ANALISIS DESCRIPTIVO GENERAL   ----------------------------#

# Definimos nuestras variables numericas y discretas para describir y para analizar posteriormente

# Variables numéricas que usaremos como predictores
numeric_features = ["home_points_before", "away_points_before", "home_form_last5", "away_form_last5",
                    "home_attack_last5", "away_attack_last5", "home_defense_last5", "away_defense_last5", 
                    "eff_on_target_home_last3", "eff_on_target_away_last3", "eff_on_target_home_last10", 
                    "eff_on_target_away_last10", 'elo_diff_before']

# Variables categóricas (equipos - local/visita - y temporada)
categorical_features = ['season', 'hometeam', 'awayteam']


# Variables numéricas relevantes para el análisis
# Guardamos la tabla
tabla_desc = df[numeric_features].describe().T.round(2)
print(tabla_desc)
# Exportamos una tabla con formato 
with open(ruta_graficos/'tabla_estadistica_descriptiva.tex', 'w', encoding='utf-8') as f:
    f.write(tabla_desc.to_latex(index=True,
                                float_format="%.2f",
                                bold_rows=True,
                                caption="Estadísticas descriptivas de las variables utilizadas en el modelo",
                                label="tab:desc_general"))


# Vamos a realizar una estandirización para mejorar la visual del boxplot
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[numeric_features] = scaler.fit_transform(df[numeric_features])

plt.figure(figsize=(12,8))
sns.boxplot(data=df_scaled[numeric_features], orient='h', palette='coolwarm')
plt.title('')
plt.xlabel('Valor estandarizado (z-score)')
plt.tight_layout()
plt.savefig(ruta_graficos/"boxplot_estadisticas_partido_creadas.png", dpi=300, bbox_inches="tight")
plt.show()

## Gráfico de distribución de los targets
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# Distribución de la diferencia de goles (target de regresión)
sns.histplot(df['goal_diff'], kde=True, bins=range(-10, 11), ax=axes[0])
axes[0].set_title('Distribución de la diferencia de goles (goal_diff)')
axes[0].set_xlabel('goal_diff')
axes[0].set_ylabel('Frecuencia')

# Distribución del target binario: gana local vs no gana
sns.countplot(x=df['home_win_bin'], ax=axes[1])
axes[1].set_title('Resultado binario: victoria local')
axes[1].set_xlabel('home_win_bin (1 = gana local, 0 = no gana)')
axes[1].set_ylabel('Frecuencia')

plt.tight_layout()
plt.savefig(ruta_graficos/"distribucion_targets.png",
            dpi=300, bbox_inches="tight")
plt.show()


#### Gráfico de correlaciones ####
corr = df[numeric_features].corr(method='spearman')

plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0,
            linewidths=.5, cbar_kws={'label': 'ρ Spearman'})
plt.title('')
plt.tight_layout()
plt.savefig(ruta_graficos/"heatmap_correlaciones.png", dpi=300, bbox_inches="tight")
plt.show()

# Correlación solo con el target goal_diff
target = 'goal_diff'
corr_complete = numeric_features + [target]
corr_target = df[corr_complete].corr(method='spearman')[[target]].sort_values(by=target, ascending=False)
plt.figure(figsize=(4, 6))
sns.heatmap(corr_target, annot=True, fmt=".2f", cmap='coolwarm', center=0, cbar=False)
plt.title('')
plt.ylabel('Variables')
plt.tight_layout()
plt.savefig(ruta_graficos/"corr_con_goal_diff.png", dpi=300, bbox_inches="tight")
plt.show()



#---------------------------  REGRESION LINEAL LASSO   ----------------------------#

# Volvemos a definir las variables pero incluimos ahoras las dummy que habias sacado para los graficos
numeric_features = ["home_points_before", "away_points_before", "home_form_last5", "away_form_last5",
                    "home_attack_last5", "away_attack_last5", "home_defense_last5", "away_defense_last5", 
                    "eff_on_target_home_last3", "eff_on_target_away_last3", "eff_on_target_home_last10", 
                    "eff_on_target_away_last10", 'elo_diff_before']


# Definimos nuestro Target
y = df['goal_diff']

# Modificamos el dataset y dejamos solo las columnas del modelo
X = df[numeric_features + categorical_features]

# Por seguridad: eliminamos filas con NA en X o y
mask = X.notna().all(axis=1) & y.notna()
X = X[mask]
y = y[mask]
print("Dimensiones después de eliminar NA:", X.shape)

# Aplicamos un preprocesamiento para corregir posibles problemas 
# Imputamos la mediana en los datos NaN y estandarizamos las variables
numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                      ('scaler', StandardScaler())])

# Corregimos las variables categóricas (dado que son texto) 
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                          ('encoder', OneHotEncoder(handle_unknown='ignore'))])

# Combinamos todo en un solo ColumnTransformer
preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Separamos los datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)
print("Train:", X_train.shape, "Test:", X_test.shape)

# Modelo LASSO. vamos a probar distintos valores para ver los resultados y compararlos
for k in [3, 5, 10]:
    lasso_cv_k = LassoCV(alphas=None, cv=k, random_state=42, max_iter=100000)
    
    lasso_pipeline_k = Pipeline(steps=[('preprocess', preprocessor), ('model', lasso_cv_k)])
    lasso_pipeline_k.fit(X_train, y_train)
    
    best_alpha_k = lasso_pipeline_k.named_steps['model'].alpha_
    print(f"cv={k} → alpha óptimo = {best_alpha_k:.5f}")

# Tomamos el valor 5 
lasso_cv = LassoCV(alphas=None,cv=5, random_state=42, max_iter=100000)
modelo_final = Pipeline(steps=[('preprocess', preprocessor),('model', lasso_cv)])

# Último entrenamiento 
modelo_final.fit(X_train, y_train)

# Predicciones
y_pred_train = modelo_final.predict(X_train)

# Métricas del entrenamiento
mse_train = mean_squared_error(y_train, y_pred_train)
rmse_train = np.sqrt(mse_train)
mae_train = mean_absolute_error(y_train, y_pred_train)
r2_train  = r2_score(y_train, y_pred_train)
print("=== MÉTRICAS TRAIN ===")
print(f"MSE  : {mse_train:.3f}")
print(f"RMSE : {rmse_train:.3f}")
print(f"MAE  : {mae_train:.3f}")
print(f"R²   : {r2_train:.3f}")

# Métricas predichas
y_pred_test  = modelo_final.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred_test)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test  = r2_score(y_test, y_pred_test)
print("\n=== MÉTRICAS TEST ===")
print(f"MSE  : {mse_test:.3f}")
print(f"RMSE : {rmse_test:.3f}")
print(f"MAE  : {mae_test:.3f}")
print(f"R²   : {r2_test:.3f}")

# Gráfico de Prediccion 
# Predicciones del modelo final (el de LassoCV)
y_pred = modelo_final.predict(X_test)
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
# Línea y = x
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
plt.xlabel('Goal diff real (y_test)')
plt.ylabel('Goal diff predicho (y_pred)')
plt.title('')
plt.savefig(ruta_graficos/"grafico_LASSO.png", dpi=300, bbox_inches="tight")
plt.tight_layout()
plt.show()



#----------------------------- EVALUACION DE VALORES ALFA -------------------------------#
# Valor óptimo encontrado por LassoCV
mejor_alpha = modelo_final.named_steps['model'].alpha_
print(f"Alpha óptimo encontrado por CV: {mejor_alpha:.5f}")

# Probamos 3 valores: menor, igual y mayor al óptimo
alphas_to_try = [mejor_alpha/5, mejor_alpha, mejor_alpha*5]

resultados_lasso = []

for a in alphas_to_try:
    # Definimos un modelo LASSO con alpha fijo
    lasso_fixed = Lasso(alpha=a, max_iter=100000, random_state=42)
    pipe_fixed = Pipeline(steps=[('preprocess', preprocessor),
                                 ('model', lasso_fixed)])
    pipe_fixed.fit(X_train, y_train)
    y_pred_a = pipe_fixed.predict(X_test)

    mse_a = mean_squared_error(y_test, y_pred_a)
    rmse_a = np.sqrt(mse_a)
    mae_a = mean_absolute_error(y_test, y_pred_a)
    r2_a  = r2_score(y_test, y_pred_a)
    resultados_lasso.append({'alpha': a, 'MSE': mse_a, 'RMSE': rmse_a, 'MAE': mae_a, 'R2': r2_a})

df_resultados_lasso = pd.DataFrame(resultados_lasso)
print(df_resultados_lasso)

# Exportamos una tabla con formato 
with open(ruta_graficos/'tabla_resultados_evaluacion_LASSO.tex', 'w', encoding='utf-8') as f:
    f.write(df_resultados_lasso.to_latex(index=True,
                                float_format="%.2f",
                                bold_rows=True,
                                caption="Evaluación de Alpha en LASSO",
                                label="tab:desc_general"))



#-------------------------- ANALISIS DE LAS VARIABLES PERTINENTES -------------------------------#

# Nombres de variables después del preprocesamiento (num + dummies)
feature_var = modelo_final.named_steps['preprocess'].get_feature_names_out()

# Coeficientes del modelo LASSO
coefs = modelo_final.named_steps['model'].coef_

importancia_lasso = (pd.DataFrame({'variable': feature_var,
                                   'coeficiente': coefs})
                                   .assign(abs_coef=lambda df: df['coeficiente'].abs()).sort_values('abs_coef', ascending=False))

#Revisamos las 20 principales variables que explican el modelo 
print(importancia_lasso.head(20)) 

# Lo presentamos en un grafico para el informe (sólo 20)
top_n = 20
top_vars = importancia_lasso.head(top_n)
plt.figure(figsize=(10, 6))
plt.barh(top_vars['variable'], top_vars['abs_coef'])
plt.gca().invert_yaxis()  # para que la más importante quede arriba
plt.xlabel('|coeficiente LASSO|')
plt.title('')
plt.tight_layout()
plt.savefig(ruta_graficos/"relevancia_variables.png", dpi=300, bbox_inches="tight")
plt.show()



#----------------------------- RANDOM FOREST -------------------------------#
# Target para clasificación
y_rf = df['home_win_bin']

# Matriz de características (mismas que en LASSO)
X_rf = df[numeric_features + categorical_features].copy()

# Target de clasificación
y_rf = df['home_win_bin']

# Eliminamos posibles NAs por seguridad
mask = X_rf.notna().all(axis=1) & y_rf.notna()
X_rf = X_rf[mask]
y_rf = y_rf[mask]

# Dividimos nuestrso datos en un set para entrenamiento y prueba
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42, stratify=y_rf)

# Revisamos el tamaño de los datset de entrenamiento y prueba
print("Tamaño train:", X_train_rf.shape)
print("Tamaño test :", X_test_rf.shape)

# Random Forest base (ajustaremos hiperparámetros después)
rf_clf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)

rf_pipeline = Pipeline(steps=[('preprocess', preprocessor), ('model', rf_clf)])

# Entrenamos
rf_pipeline.fit(X_train_rf, y_train_rf)

# Predicción en test
y_pred_rf = rf_pipeline.predict(X_test_rf)

print("Accuracy en test:", accuracy_score(y_test_rf, y_pred_rf))
print("\nClassification report:\n", classification_report(y_test_rf, y_pred_rf))
print("\nMatriz de confusión:\n", confusion_matrix(y_test_rf, y_pred_rf))


# Matriz de confusión gráfico 
cm = confusion_matrix(y_test_rf, y_pred_rf)

# Gráfico estilizado
plt.figure(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=['No gana local (0)', 'Gana local (1)'])
disp.plot(cmap='Blues', values_format='d')
plt.title("")
plt.tight_layout()
plt.savefig(ruta_graficos/"matriz_confusion_random_forest.png",
            dpi=300, bbox_inches="tight")
plt.show()








## ANEXO 

# SE CREARON NUEVAS VARIABLES QUE TUVIESEN UNA MAYOR CORRELACION CON EL TARGET, AL COMPROBAR QUE NO AFECTAN LA CALIDAD
# DEL MODELO, SE DECIDIÓ SÓLO INCLUIRLAS EN EL CÓDIGO PERO NO EN EL ANÁLISIS DEL INFORME FINAL. ASIMISMO, SE INCORPORÓ
# LAS VARIABLES DE GOLES EN EL PRIMER TIEMPO, QUE TAMPOCO SE INCLUYERON EN EL INFORME FINAL.

# Ordenamos datos cronológicamente
df = df.sort_values('matchdate')
# Creamos una columna para identificar cada enfrentamiento único
df['pair'] = df.apply(lambda x: tuple(sorted([x['hometeam'], x['awayteam']])), axis=1)
# Creamos estructura para almacenar último diff previo
último_diff = {}
h2h_goal_diff_last = []
for i, row in df.iterrows():
    equipo_local = row['hometeam']
    equipo_visita = row['awayteam']
    pair = row['pair']
    # si existe historial, agregamos
    if pair in último_diff:
        h2h_goal_diff_last.append(último_diff[pair])
    else:
        h2h_goal_diff_last.append(0)  # sin historial
    
    # actualizamos historial para próximos partidos
    nuevo_diff = row['fulltimehomegoals'] - row['fulltimeawaygoals']
    último_diff[pair] = nuevo_diff

# añadimos la variable
df['h2h_goal_diff_last'] = h2h_goal_diff_last



#---------------------------  REGRESION LINEAL LASSO   ----------------------------#

# Volvemos a definir las variables pero incluimos ahoras las dummy que habias sacado para los graficos
numeric_features = ["home_points_before", "away_points_before", "home_form_last5", "away_form_last5",
                    "home_attack_last5", "away_attack_last5", "home_defense_last5", "away_defense_last5", 
                    "eff_on_target_home_last3", "eff_on_target_away_last3", "eff_on_target_home_last10", 
                    "eff_on_target_away_last10", 'elo_diff_before', 'halftimehomegoals', 'halftimeawaygoals',
                    'h2h_goal_diff_last']


# Definimos nuestro Target
y = df['goal_diff']

# Modificamos el dataset y dejamos solo las columnas del modelo
X = df[numeric_features + categorical_features]

# Por seguridad: eliminamos filas con NA en X o y
mask = X.notna().all(axis=1) & y.notna()
X = X[mask]
y = y[mask]
print("Dimensiones después de eliminar NA:", X.shape)

# Aplicamos un preprocesamiento para corregir posibles problemas 
# Imputamos la mediana en los datos NaN y estandarizamos las variables
numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                      ('scaler', StandardScaler())])

# Corregimos las variables categóricas (dado que son texto) 
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                          ('encoder', OneHotEncoder(handle_unknown='ignore'))])

# Combinamos todo en un solo ColumnTransformer
preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Separamos los datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)
print("Train:", X_train.shape, "Test:", X_test.shape)

# Modelo LASSO. vamos a probar distintos valores para ver los resultados y compararlos
for k in [3, 5, 10]:
    lasso_cv_k = LassoCV(alphas=None, cv=k, random_state=42, max_iter=100000)
    
    lasso_pipeline_k = Pipeline(steps=[('preprocess', preprocessor), ('model', lasso_cv_k)])
    lasso_pipeline_k.fit(X_train, y_train)
    
    best_alpha_k = lasso_pipeline_k.named_steps['model'].alpha_
    print(f"cv={k} → alpha óptimo = {best_alpha_k:.5f}")

# Tomamos el valor 5 
lasso_cv = LassoCV(alphas=None,cv=5, random_state=42, max_iter=100000)
modelo_final = Pipeline(steps=[('preprocess', preprocessor),('model', lasso_cv)])

# Último entrenamiento 
modelo_final.fit(X_train, y_train)

# Predicciones
y_pred_train = modelo_final.predict(X_train)

# Métricas del entrenamiento
mse_train = mean_squared_error(y_train, y_pred_train)
rmse_train = np.sqrt(mse_train)
mae_train = mean_absolute_error(y_train, y_pred_train)
r2_train  = r2_score(y_train, y_pred_train)
print("=== MÉTRICAS TRAIN ===")
print(f"MSE  : {mse_train:.3f}")
print(f"RMSE : {rmse_train:.3f}")
print(f"MAE  : {mae_train:.3f}")
print(f"R²   : {r2_train:.3f}")

# Métricas predichas
y_pred_test  = modelo_final.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred_test)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test  = r2_score(y_test, y_pred_test)
print("\n=== MÉTRICAS TEST ===")
print(f"MSE  : {mse_test:.3f}")
print(f"RMSE : {rmse_test:.3f}")
print(f"MAE  : {mae_test:.3f}")
print(f"R²   : {r2_test:.3f}")

# Gráfico de Prediccion 
# Predicciones del modelo final (el de LassoCV)
y_pred = modelo_final.predict(X_test)
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
# Línea y = x
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
plt.xlabel('Goal diff real (y_test)')
plt.ylabel('Goal diff predicho (y_pred)')
plt.title('')
plt.tight_layout()
plt.show()



#----------------------------- RANDOM FOREST -------------------------------#
# Target para clasificación
y_rf = df['home_win_bin']

# Matriz de características (mismas que en LASSO)
X_rf = df[numeric_features + categorical_features].copy()

# Target de clasificación
y_rf = df['home_win_bin']

# Eliminamos posibles NAs por seguridad
mask = X_rf.notna().all(axis=1) & y_rf.notna()
X_rf = X_rf[mask]
y_rf = y_rf[mask]

# Dividimos nuestrso datos en un set para entrenamiento y prueba
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42, stratify=y_rf)

# Revisamos el tamaño de los datset de entrenamiento y prueba
print("Tamaño train:", X_train_rf.shape)
print("Tamaño test :", X_test_rf.shape)

# Random Forest base (ajustaremos hiperparámetros después)
rf_clf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)

rf_pipeline = Pipeline(steps=[('preprocess', preprocessor), ('model', rf_clf)])

# Entrenamos
rf_pipeline.fit(X_train_rf, y_train_rf)

# Predicción en test
y_pred_rf = rf_pipeline.predict(X_test_rf)

print("Accuracy en test:", accuracy_score(y_test_rf, y_pred_rf))
print("\nClassification report:\n", classification_report(y_test_rf, y_pred_rf))
print("\nMatriz de confusión:\n", confusion_matrix(y_test_rf, y_pred_rf))


# Matriz de confusión gráfico 
cm = confusion_matrix(y_test_rf, y_pred_rf)

# Gráfico estilizado
plt.figure(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=['No gana local (0)', 'Gana local (1)'])
disp.plot(cmap='Blues', values_format='d')
plt.title("")
plt.tight_layout()
plt.show()
