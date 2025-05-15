import pandas as pd
df = pd.read_csv('bootcamp_train (1).csv')


#Mudança e tratamento inicial de dados
"""Remoção de linhas poluídas com muitos erros"""

# Lista dos índices das linhas que você quer remover (as 35 linhas com mais de 2 valores nulos)
indices_para_remover = [32, 101, 169, 250, 393, 403, 450, 487, 538, 548, 572, 762, 1000, 1022, 1084, 1256, 1392, 1431, 1450, 1491, 1619, 1697, 1837, 1883, 1887, 1908, 2037, 2315, 2455, 2889, 2921, 3052, 3227, 3247, 3263]

# Remove as linhas com os índices especificados
df_sem_linhas_nulos = df.drop(index=indices_para_remover)

# Se você quiser substituir o DataFrame original:
df = df_sem_linhas_nulos




"""Normalizando falhas para binario 1 e 0"""

mapa_falha = {
    'False': 0, '0': 0, 'Não': 0, 'nao': 0, 'N': 0, 'n': 0, False: 0,
    'True': 1, '1': 1, 'Sim': 1, 'sim': 1, 'S': 1, 'y': 1, True: 1
}

# Lista das colunas de falha conforme informado
falhas = [
    'falha_1', 'falha_2', 'falha_3',
    'falha_4', 'falha_5', 'falha_6',
    'falha_outros'
]

# Aplicar a conversão para cada coluna de falha
for col in falhas:
    df[col] = df[col].astype(str).str.strip().map(mapa_falha)
    # Substituir quaisquer NaN resultantes por 0
    df[col].fillna(0, inplace=True)
    # Converter para inteiro
    df[col] = df[col].astype(int)

# Verificar se a normalização foi bem-sucedida
print(df[falhas].head())





"""Normalizando os valores da coluna categorica tipo de aço 300 e 400 para 1 e 0 para evitar disparidades e erros no modelo."""
#Conversão de colunas binárias (tipo de aço)
# Preenche valores ausentes pela moda e mapeia strings para 0/1
import pandas as pd
import numpy as np

mapping_a300 = {
    'não': 0.0, 'nao': 0.0, 'n': 0.0, '0': 0.0, '-': np.nan,  # '-' será tratado como nulo para imputação
    'sim': 1.0, 's': 1.0, '1': 1.0, 'yes': 1.0, 'y': 1.0, 'sim': 1.0, 'Sim': 1.0, 'Não': 0.0, 'N': 0.0
}

mapping_a400 = {
    'não': 0.0, 'nao': 0.0, 'n': 0.0, '0': 0.0, 'nao': 0.0, 'nan': np.nan, # 'nan' será tratado como nulo
    'sim': 1.0, 's': 1.0, '1': 1.0, 'yes': 1.0, 'y': 1.0, 'sim': 1.0, 'Sim': 1.0, 'Não': 0.0, 'S': 1.0
}

df['tipo_do_aço_A300'] = df['tipo_do_aço_A300'].astype(str).str.lower().map(mapping_a300).astype(float)
df['tipo_do_aço_A400'] = df['tipo_do_aço_A400'].astype(str).str.lower().map(mapping_a400).astype(float)

# Verificar se ainda há nulos após o mapeamento
print(df[['tipo_do_aço_A300', 'tipo_do_aço_A400']].isnull().sum())

"""Preenchendo nulos com moda

"""

moda_a400 = df['tipo_do_aço_A400'].mode()[0]
df['tipo_do_aço_A400'].fillna(moda_a400, inplace=True)
moda_a300 = df['tipo_do_aço_A300'].mode()[0]
df['tipo_do_aço_A300'].fillna(moda_a300, inplace=True)

for col in ['tipo_do_aço_A300', 'tipo_do_aço_A400']:
    print(f"Valores únicos na coluna '{col}': {df[col].unique()}")






#Tratamento de nulos numericos
"""Trocando nulos por  mediana para categoria numerica

"""

numeric_cols = [
    'x_maximo',
    'soma_da_luminosidade',
    'maximo_da_luminosidade',
    'espessura_da_chapa_de_aço',
    'index_quadrado',
    'indice_global_externo',
    'indice_de_luminosidade'
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    #df[col].fillna(0, inplace=True)
    df[col].fillna(df[col].median(), inplace=True)


#Tratamento
"""Retirar id pois é repetido e desnecessáio. E peso da placa pois tem apenas um valor que seria uma constante e uma reta que nao influenciaira nenhum."""

df.drop(columns=['id', 'peso_da_placa'], inplace=True)


#Tratamento

"""Tornei as colunas positivas, coorigindo os valores

"""

cols_to_correct = ['x_minimo',	'x_maximo',	'y_minimo',	'y_maximo',	'area_pixels',	'perimetro_x',	'perimetro_y', 	'soma_da_luminosidade',	'maximo_da_luminosidade',	'comprimento_do_transportador',	'espessura_da_chapa_de_aço',	'temperatura',	'index_de_bordas',	'index_vazio',	'index_quadrado',	'index_externo_x', 	'indice_de_bordas_x',	'indice_de_bordas_y',	'indice_de_variacao_x', 	'indice_de_variacao_y',	'indice_global_externo',	'log_das_areas',	'log_indice_x',	'log_indice_y',	'indice_de_orientaçao',	'indice_de_luminosidade',	'sigmoide_das_areas',	'minimo_da_luminosidade']


for col in cols_to_correct:
    df[col] = np.abs(df[col])

#Metodo multiclasse
# Importando bibliotecas necessárias
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Definindo as colunas de falhas
falhas = ['falha_1', 'falha_2', 'falha_3', 'falha_4', 'falha_5', 'falha_6', 'falha_outros']

# Convertendo as falhas em uma coluna única
label_map = {name: idx for idx, name in enumerate(falhas)}
df['y_multiclass'] = df[falhas].idxmax(axis=1).map(label_map)

# Garantindo que os labels estejam em formato numérico
label_encoder = LabelEncoder()
df['y_multiclass'] = label_encoder.fit_transform(df['y_multiclass'])

# Definindo X e y
X = df.drop(columns=falhas + ['y_multiclass'])
y = df['y_multiclass']

# Divisão treino/teste com stratify
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Pipeline de pré-processamento
numerical = X.select_dtypes(include=['number']).columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical)
    ]
)

# Definindo os modelos corrigidos
models = {
    'RandomForest': Pipeline([
        ('preprocessing', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
    ]),
    'XGBoost': Pipeline([
        ('preprocessing', preprocessor),
        ('classifier', XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', random_state=42))
    ]),
    'SVM': Pipeline([
        ('preprocessing', preprocessor),
        ('classifier', SVC(probability=True, random_state=42))
    ]),
    'KNN': Pipeline([
        ('preprocessing', preprocessor),
        ('classifier', KNeighborsClassifier())
    ]),
    'LogisticRegression': Pipeline([
        ('preprocessing', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
    ]),
    'DecisionTree': Pipeline([
        ('preprocessing', preprocessor),
        ('classifier', DecisionTreeClassifier(random_state=42, class_weight='balanced'))
    ])
}

# Ensemble (Voting Classifier) - Soft Voting para probabilidades
ensemble = VotingClassifier(
    estimators=[
        ('rf', models['RandomForest']),
        ('xgb', models['XGBoost']),
        ('svc', models['SVM'])
    ],
    voting='soft'
)
models['Ensemble'] = ensemble

# Treinamento e avaliação
for name, model in models.items():
    print(f"\n*** Treinando e Avaliando Modelo: {name} ***")
    model.fit(X_train, y_train)

    # Previsão
    y_pred = model.predict(X_test)

    # Classificação
    print(f"\n-- Métricas para {name} --")
    print(classification_report(y_test, y_pred, target_names=[str(cls) for cls in label_encoder.classes_]))


    # Matriz de Confusão
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(cmap='Blues')
    plt.title(f"Matriz de Confusão - {name}")
    plt.show()

    # ROC AUC Macro
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)
        y_test_bin = pd.get_dummies(y_test)
        auc_macro = roc_auc_score(y_test_bin, y_proba, average='macro', multi_class='ovr')
        print(f"ROC AUC (Macro) para {name}: {auc_macro:.4f}")

    # Validação Cruzada
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"Validação Cruzada (Acurácia) para {name}: {cv_scores}")
    print(f"Acurácia Média: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

#Treino do xgboost modelo escolhido e de teste de um exemplo
# Importando bibliotecas necessárias
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

# Definindo as colunas de falhas
falhas = ['falha_1', 'falha_2', 'falha_3', 'falha_4', 'falha_5', 'falha_6', 'falha_outros']

# Convertendo as falhas em uma coluna única
label_map = {name: idx for idx, name in enumerate(falhas)}
df['y_multiclass'] = df[falhas].idxmax(axis=1).map(label_map)

# Garantindo que os labels estejam em formato numérico
label_encoder = LabelEncoder()
df['y_multiclass'] = label_encoder.fit_transform(df['y_multiclass'])

# Definindo X e y
X = df.drop(columns=falhas + ['y_multiclass'])
y = df['y_multiclass']

# Divisão treino/teste com stratify
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Pipeline de pré-processamento
numerical = X.select_dtypes(include=['number']).columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical)
    ]
)

# Modelo XGBoost com pipeline
xgb_clf = Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', random_state=42))
])

# Treinamento
print("\n*** Treinando e Avaliando Modelo: XGBoost ***")
xgb_clf.fit(X_train, y_train)

# Previsão
y_pred = xgb_clf.predict(X_test)
y_proba = xgb_clf.predict_proba(X_test)

# Avaliação
print("\n-- Métricas para XGBoost --")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_.astype(str)))

# Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap='Blues')
plt.title("Matriz de Confusão - XGBoost")
plt.tight_layout()
plt.show()

# ROC AUC (métricas multiclasse)
roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
print(f"ROC AUC (Macro) para XGBoost: {roc_auc:.4f}")

# Validação cruzada
cv_scores = cross_val_score(xgb_clf, X, y, cv=10, scoring='accuracy')
print(f"\nValidação Cruzada (Acurácia) para XGBoost: {cv_scores}")
print(f"Acurácia Média: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

# Previsão com nova imagem (exemplo)
nova_imagem = pd.DataFrame([{
    'id': 1,
    'x_minimo': 10.5,
    'x_maximo': 30.2,
    'y_minimo': 5.1,
    'y_maximo': 25.3,
    'peso_da_placa': 100,
    'area_pixels': 345,
    'perimetro_x': 50.6,
    'perimetro_y': 48.2,
    'soma_da_luminosidade': 2300,
    'maximo_da_luminosidade': 255,
    'comprimento_do_transportador': 100.5,
    'tipo_do_aço_A300': 1,
    'tipo_do_aço_A400': 0,
    'espessura_da_chapa_de_aço': 4.5,
    'temperatura': 78.3,
    'index_de_bordas': 0.3,
    'index_vazio': 0.2,
    'index_quadrado': 0.1,
    'index_externo_x': 1.1,
    'indice_de_bordas_x': 0.7,
    'indice_de_bordas_y': 0.5,
    'indice_de_variacao_x': 0.05,
    'indice_de_variacao_y': 0.04,
    'indice_global_externo': 0.5,
    'log_das_areas': 1.5,
    'log_indice_x': 0.8,
    'log_indice_y': 0.3,
    'indice_de_orientaçao': 0.4,
    'indice_de_luminosidade': 0.9,
    'sigmoide_das_areas': 0.8,
    'minimo_da_luminosidade': 50
}])


# Garantir que a nova amostra tenha as mesmas colunas do treino
colunas_treino = X.columns
for coluna in colunas_treino:
    if coluna not in nova_imagem.columns:
        nova_imagem[coluna] = 0

# Reordenar as colunas
nova_imagem = nova_imagem[colunas_treino]

# Pré-processar a nova imagem usando o pipeline completo
nova_imagem_scaled = xgb_clf.named_steps['preprocessing'].transform(nova_imagem)

# Converter de volta para DataFrame com as colunas corretas
nova_imagem_scaled = pd.DataFrame(nova_imagem_scaled, columns=colunas_treino)

# Prever a classe e a probabilidade com o modelo XGBoost
pred_classe = xgb_clf.predict(nova_imagem_scaled)
pred_proba = xgb_clf.predict_proba(nova_imagem_scaled)

# Resultado da predição
print(f"\nClasse predita: {label_encoder.inverse_transform([pred_classe[0]])[0]}")
print(f"Probabilidades por classe: {pred_proba[0]}")

# Exibir a classe com maior probabilidade
classe_predita = np.argmax(pred_proba, axis=1)
probabilidade = np.max(pred_proba, axis=1)
print(f"Classe Predita: {label_encoder.inverse_transform([classe_predita[0]])[0]} com Probabilidade: {probabilidade[0]:.4f}")

