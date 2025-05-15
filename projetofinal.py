# -*- coding: utf-8 -*-


from google.colab import files
uploaded = files.upload()

import pandas as pd
df = pd.read_csv('bootcamp_train (1).csv')

#Inicio da analise dos dados

#Identificar o tamanho da tabela
df.shape


#Verificar e analisar a presença de nulos em cada coluna
df.isnull().sum()

#Verficar os valore unicos de cada coluna
df.nunique()


#Verificar a presença de duplicatas
df[df.duplicated(keep=False)]

df[df.duplicated(subset=['x_minimo', 'x_maximo', 'y_minimo', 'y_maximo','soma_da_luminosidade','temperatura'], keep=False)]



numerical = [col for col in df.columns if df[col].nunique() > 10]
categorical = df.columns.difference(numerical).to_list()

#Fução para analisar estattistica desscritiva alem dos que tem a função .describe

import pandas as pd

def analisar_coluna_numerica(serie):
    """
    Calcula e exibe estatísticas descritivas de uma Series numérica do Pandas.

    Args:
        serie (pd.Series): A coluna numérica (Series) a ser analisada.
    """
    print(f"Análise da coluna numérica: '{serie.name}'")
    print(f"- Número de valores não nulos: {serie.count()}")
    print(f"- Número de valores nulos: {serie.isnull().sum()}")
    print(f"- Tipo de dados: {serie.dtype}")
    print("\nMedidas de Tendência Central:")
    print(f"  - Média: {serie.mean():.2f}")
    print(f"  - Mediana: {serie.median():.2f}")
    moda = serie.mode()
    if not moda.empty:
        print(f"  - Moda: {moda.tolist()}") 
    else:
        print("  - Moda: Não há moda única")
    print("\nMedidas de Dispersão:")
    print(f"  - Desvio Padrão: {serie.std():.2f}")
    print(f"  - Variância: {serie.var():.2f}")
    print(f"  - Amplitude (máximo - mínimo): {serie.max() - serie.min():.2f}")
    print("\nMedidas de Forma:")
    print(f"  - Assimetria (Skewness): {serie.skew():.2f}")
    print(f"  - Curtose (Kurtosis): {serie.kurt():.2f}")
    print("\nPercentis:")
    print(f"  - Percentil 25 (Q1): {serie.quantile(0.25):.2f}")
    print(f"  - Percentil 50 (Mediana): {serie.quantile(0.50):.2f}")
    print(f"  - Percentil 75 (Q3): {serie.quantile(0.75):.2f}")
    print(f"  - Mínimo: {serie.min():.2f}")
    print(f"  - Máximo: {serie.max():.2f}")
    print("-" * 30)


df_numeric = df.select_dtypes(include=['number'])
for coluna in df_numeric.columns:
    analisar_coluna_numerica(df_numeric[coluna])



"""Plotagem de histogramas"""

import matplotlib.pyplot as plt
import seaborn as sns
data= df
def plotar_frequencia(data):
    for coluna in data.columns:
        plt.figure(figsize=(8, 5))

        
        if data[coluna].dtype == 'object' or data[coluna].nunique() < 20:
            # Gráfico de barras para variáveis categóricas ou com poucos valores únicos
            sns.countplot(data=data, x=coluna, palette='viridis')
            plt.title(f'Frequência da variável {coluna}')
            plt.xlabel(coluna)
            plt.ylabel('Frequência')

        else:
            # Histograma para variáveis numéricas
            sns.histplot(data=data, x=coluna, kde=True, bins=20, color='teal')
            plt.title(f'Distribuição da variável {coluna}')
            plt.xlabel(coluna)
            plt.ylabel('Frequência')

        plt.xticks(rotation=45)  
        plt.tight_layout()
        plt.show()


plotar_frequencia(data)

for column in df.columns:
    # Verifica se a coluna é numérica para criar o boxplot
    if pd.api.types.is_numeric_dtype(df[column]):
        # Cria uma nova figura para cada boxplot
        plt.figure(figsize=(8, 6))
        # Cria o boxplot para a coluna atual
        df.boxplot(column=column)
        # Define o título do gráfico
        plt.title(f'Boxplot da Coluna: {column}')
        # Exibe o gráfico
        plt.show()
    else:
        print(f"A coluna '{column}' não é numérica e o boxplot não pode ser gerado.")





"""Plotagem de matriz de correlaçao"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# 1. Selecionar apenas as colunas numéricas para a matriz de correlação
df_numeric = df.select_dtypes(include=['number'])

# 2. Calcular a matriz de correlação
correlation_matrix = df_numeric.corr()

# 3. Criar o mapa de calor (heatmap) da matriz de correlação
plt.figure(figsize=(12, 10))  # Define o tamanho da figura
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Matriz de Correlação das Colunas Numéricas')
plt.show()




import pandas as pd
import numpy as np
df_numeric = df.select_dtypes(include=['number'])
correlation_matrix = df_numeric.corr()

# 1. Extrair os valores da matriz de correlação (excluindo a diagonal e duplicatas)
upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
correlation_values = upper_triangle.unstack().dropna().sort_values(ascending=False)

print("Valores de correlação (em ordem decrescente):")
display(correlation_values.head())
for index, value in correlation_values.head(2000).items():
    print(f"{index}: {value}")
# 2. Imprimir os valores de correlação em ordem decrescente

skewness_todas = df.select_dtypes(include=['number']).skew()
print("\nSkewness de todas as colunas numéricas:\n", skewness_todas)

#Plotagem de grafico de pizza das falhas
falhas = ['falha_1', 'falha_2', 'falha_3', 'falha_4', 'falha_5', 'falha_6', 'falha_outros']
import pandas as pd
import matplotlib.pyplot as plt

# Calcula a frequência de ocorrência de cada tipo de falha
frequencia_falhas = df[falhas].sum().sort_values(ascending=False)

# Cria o gráfico de pizza
plt.figure(figsize=(8, 8))  # Define o tamanho da figura
plt.pie(frequencia_falhas, labels=frequencia_falhas.index, autopct='%1.1f%%', startangle=90, colors=['gold', 'lightcoral', 'lightskyblue', 'lightgreen', 'orange', 'purple', 'pink'])
plt.title('Distribuição dos Tipos de Falha')  # Adiciona um título ao gráfico
plt.axis('equal')  # Garante que o gráfico seja um círculo
plt.show()



#Gráfico contagem de valores negativos por coluna
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Lista de todas as colunas do DataFrame
todas_colunas = df.columns

# Filtra apenas as colunas numéricas
numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

# Calcula a contagem de valores negativos para cada coluna
contagem_negativos = (df[numeric_columns] < 0).sum()

# Filtra as colunas que têm pelo menos um valor negativo
colunas_com_negativos = contagem_negativos[contagem_negativos > 0]

# Cria o gráfico de barras apenas com as colunas que têm negativos
plt.figure(figsize=(15, 7))  # Aumenta o tamanho da figura para melhor visualização com muitas colunas
colunas_com_negativos.plot(kind='bar', color='skyblue')
plt.title('Contagem de Valores Negativos por Coluna (Apenas Colunas Numéricas com Negativos)')
plt.xlabel('Colunas')
plt.ylabel('Contagem de Valores Negativos')
plt.xticks(rotation=45, ha='right') 
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


#Gráfico de contagem de outliers por coluna

# Lista de todas as colunas do DataFrame
todas_colunas = df.columns

# Filtra apenas as colunas numéricas
numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

def calcular_outliers(s):

    Q1 = s.quantile(0.25)
    Q3 = s.quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    return (s < limite_inferior) | (s > limite_superior)

# Calcula a contagem de outliers para cada coluna numérica
contagem_outliers = df[numeric_columns].apply(calcular_outliers).sum()

# Filtra as colunas que têm pelo menos um outlier
colunas_com_outliers = contagem_outliers[contagem_outliers > 0]

# Cria o gráfico de barras apenas com as colunas que têm outliers
plt.figure(figsize=(15, 7))
colunas_com_outliers.plot(kind='bar', color='coral')  # Usei uma cor diferente para destacar
plt.title('Contagem de Outliers por Coluna (Apenas Colunas Numéricas com Outliers)')
plt.xlabel('Colunas')
plt.ylabel('Contagem de Outliers')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


#Graficos de estatistica e analise multivariada

# Importando bibliotecas necessárias
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Configurações de estilo
sns.set(style="whitegrid")


### 1. Heatmap de Correlação ###
plt.figure(figsize=(16, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Heatmap de Correlação entre as Variáveis")
plt.show()

### 2. Boxplots para Mostrar Outliers ###
# Selecionando apenas as colunas numéricas
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

# Plotando os boxplots
plt.figure(figsize=(18, 12))
for i, col in enumerate(numerical_cols):
    plt.subplot((len(numerical_cols) + 2) // 3, 3, i + 1)
    sns.boxplot(data=df, x=col, color="skyblue")
    plt.title(f"Boxplot de {col}")
    plt.xlabel(col)
plt.tight_layout()
plt.show()

### 3. Gráfico de Barras para Distribuição de Valores Nulos por Coluna ###
nulos = df.isnull().sum()
nulos = nulos[nulos > 0]  # Filtrando apenas colunas com valores nulos

plt.figure(figsize=(12, 8))
sns.barplot(x=nulos.index, y=nulos.values, palette="viridis")
plt.title("Distribuição de Valores Nulos por Coluna")
plt.xlabel("Colunas")
plt.ylabel("Número de Valores Nulos")
plt.xticks(rotation=45)
plt.show()

#Histograma por categoria
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Combinar falhas em uma única coluna 'falha'
falhas = ['falha_1','falha_2','falha_3','falha_4','falha_5','falha_6','falha_outros']
df['falha'] = df[falhas].idxmax(axis=1)

# Defina a função plot_histograms
def plot_histograms(df, column, category_col, bins, custom_labels=None):

    df_plot = df.copy()

    if custom_labels and category_col in custom_labels:
        label_list = custom_labels[category_col]
        unique_values = sorted(df_plot[category_col].dropna().unique())
        label_map = {val: label_list[i] for i, val in enumerate(unique_values) if i < len(label_list)}
        df_plot[category_col] = df_plot[category_col].map(label_map)

    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=df_plot,
        x=column,
        hue=category_col,
        kde=True,
        bins=bins,
        palette='tab10',
        element='step'
    )
    plt.title(f'Histograma de {column} por {category_col}')
    plt.xlabel(column)
    plt.ylabel('Frequência')
    plt.legend(title=category_col)
    plt.tight_layout()
    plt.show()

# Lista de colunas numéricas
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

# Remover as colunas 'id' e as colunas de falha
columns_to_drop = ['id'] + falhas
numerical_cols = [col for col in numerical_cols if col not in columns_to_drop]


# Calcular o número de bins usando a raiz quadrada do número de amostras
bins = int(np.sqrt(len(df)))

# Defina os rótulos personalizados para a coluna 'falha'
custom_labels = {'falha': ['Falha 1', 'Falha 2', 'Falha 3', 'Falha 4', 'Falha 5', 'Falha 6', 'Outros']}

# Loop através das colunas numéricas e gere os histogramas
for column in numerical_cols:
    plot_histograms(df.copy(), column, 'falha', bins, custom_labels)
    

#Boxplot por categoria
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


#  Combinar falhas em uma única coluna 'falha'
falhas = ['falha_1','falha_2','falha_3','falha_4','falha_5','falha_6','falha_outros']
df['falha'] = df[falhas].idxmax(axis=1)


# Lista de colunas numéricas
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

# Remover as colunas 'id' e as colunas de falha
columns_to_drop = ['id'] + falhas
numerical_cols = [col for col in numerical_cols if col not in columns_to_drop]


# Loop através das colunas numéricas e gera os boxplots
for column in numerical_cols:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='falha', y=column, palette='viridis')
    plt.title(f'Boxplot de {column} por Falha')
    plt.xlabel('Falha')
    plt.ylabel(column)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Função para mostrar a distribuição das falhas

colunas_categoricas = falhas  

# Loop para gerar um gráfico de barras para cada coluna categórica
for coluna_categorica in colunas_categoricas:
    # Conta a frequência de cada valor na coluna categórica
    contagem_valores = df[coluna_categorica].value_counts().sort_index()

    # Imprime a contagem de valores
    print(f"Contagem de valores na coluna '{coluna_categorica}':")
    print(contagem_valores)

    # Cria o gráfico de barras
    plt.figure(figsize=(10, 6))  # Define o tamanho da figura para cada gráfico
    contagem_valores.plot(kind='bar', color='skyblue')  # Cria o gráfico de barras
    plt.title(f'Frequência de Valores na Coluna Categórica "{coluna_categorica}"')  # Define o título
    plt.xlabel('Valores')  # Define o rótulo do eixo x
    plt.ylabel('Frequência')  # Define o rótulo do eixo y
    plt.xticks(rotation=0)  # Mantém os rótulos do eixo x na horizontal
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Adiciona uma grade no eixo y
    plt.tight_layout()  # Ajusta o layout para evitar cortes
    plt.show()  # Exibe o gráfico








#Detectar os outliers e fazer a contagem de cada coluna:

import pandas as pd

def detectar_outliers_iqr(df, coluna):
    Q1 = df[coluna].quantile(0.25)
    Q3 = df[coluna].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    outliers = df[(df[coluna] < limite_inferior) | (df[coluna] > limite_superior)]
    return outliers.shape[0]  # Retorna o número de outliers

print("Contagem de outliers (IQR) por coluna numérica:")
for coluna in df.select_dtypes(include=['number']).columns:
    if df[coluna].nunique() > 10:  # Considera colunas com mais de 10 valores únicos como não categóricas
        num_outliers = detectar_outliers_iqr(df, coluna)
        print(f"Coluna '{coluna}': {num_outliers} outliers")
    else:
        print(f"Coluna '{coluna}': Provavelmente categórica (poucos valores únicos), análise de outliers ignorada.")





#Contar e mostrar os valores unicos de cada coluna
todas_colunas = df.columns

# Loop para iterar sobre cada coluna e mostrar os valores únicos e a contagem
for coluna in todas_colunas:
    # Obtém os valores únicos da coluna atual
    valores_unicos = df[coluna].unique()

    # Obtém a contagem de valores únicos da coluna atual
    contagem_valores_unicos = df[coluna].nunique()

    # Imprime o nome da coluna
    print(f"Coluna: {coluna}")

    # Imprime os valores únicos
    print(f"  Valores Únicos: {valores_unicos}")

    # Imprime a contagem de valores únicos
    print(f"  Contagem de Valores Únicos: {contagem_valores_unicos}")

    # Imprime uma linha separadora para melhor visualização
    print("-" * 30)





"""testar para saber se tem falhas ocorrem simultaneamnet ou nao, e se e multiclasse ou multirotulo"""

# Lista das colunas de falha
falhas = ['falha_1', 'falha_2', 'falha_3', 'falha_4', 'falha_5', 'falha_6', 'falha_outros']

# Soma os valores em cada linha para verificar quantos '1' existem por linha
df['soma_falhas'] = df[falhas].sum(axis=1)

# Filtra linhas com mais de uma falha ativa
linhas_multifalhas = df[df['soma_falhas'] > 1]

# Verifica se há alguma linha com múltiplas falhas
if not linhas_multifalhas.empty:
    print(f" Existem {len(linhas_multifalhas)} linhas com mais de uma falha ativa!")
    print(linhas_multifalhas)
else:
    print(" Nenhuma linha possui múltiplas falhas ativas.")

# Remove a coluna auxiliar após a verificação
df.drop(columns=['soma_falhas'], inplace=True)



#Realização da contagem de outliers por coluna para facilitar a identificação de valores distorcidos e melhorar entendimento da distribuição das variaveis e entender quem necessita ser alterado
import pandas as pd

def detectar_outliers_iqr(df, coluna):
    Q1 = df[coluna].quantile(0.25)
    Q3 = df[coluna].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    outliers = df[(df[coluna] < limite_inferior) | (df[coluna] > limite_superior)]
    return outliers.shape[0]  # Retorna o número de outliers

print("Contagem de outliers (IQR) por coluna numérica:")
for coluna in df.select_dtypes(include=['number']).columns:
    if df[coluna].nunique() > 10:  # Considera colunas com mais de 10 valores únicos como não categóricas
        num_outliers = detectar_outliers_iqr(df, coluna)
        print(f"Coluna '{coluna}': {num_outliers} outliers")
    else:
        print(f"Coluna '{coluna}': Provavelmente categórica (poucos valores únicos), análise de outliers ignorada.")










#Conta quais linhas apresentam vários outliers

import pandas as pd

def detectar_outliers_iqr_indices(df, coluna):

    Q1 = df[coluna].quantile(0.25)
    Q3 = df[coluna].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    outliers_indices = df[(df[coluna] < limite_inferior) | (df[coluna] > limite_superior)].index
    return outliers_indices

outlier_indices_por_coluna = {}

# Itera por cada coluna numérica para identificar os índices dos outliers
colunas_numericas = df.select_dtypes(include=['number']).columns
for coluna in colunas_numericas:
    if df[coluna].nunique() > 10:  # Considera colunas com mais de 10 valores únicos
        outlier_indices = detectar_outliers_iqr_indices(df, coluna)
        if not outlier_indices.empty:
            outlier_indices_por_coluna[coluna] = set(outlier_indices)

# Conta quantas vezes cada linha tem outliers em pelo menos 7 colunas
contagem_coincidencia = {}
todos_indices_de_outliers = set()
for indices in outlier_indices_por_coluna.values():
    todos_indices_de_outliers.update(indices)

for index in sorted(list(todos_indices_de_outliers)):
    colunas_neste_outlier = [coluna for coluna, indices in outlier_indices_por_coluna.items() if index in indices]
    if len(colunas_neste_outlier) >= 7:
        contagem_coincidencia[index] = colunas_neste_outlier

# Exibe a contagem de linhas com outliers em pelo menos 7 colunas
num_linhas_com_coincidencia = len(contagem_coincidencia)
print(f"Número de linhas com outliers em pelo menos 7 colunas: {num_linhas_com_coincidencia}")


if num_linhas_com_coincidencia > 0:
    print("\nÍndices das linhas com outliers em pelo menos 7 colunas:")
    print(list(contagem_coincidencia.keys()))






#Conta quais linhas apresentam mais de dois valores nulos de categorias diferentes
import pandas as pd
import numpy as np

def contar_nulos_por_linha(df):

    # Identifica colunas do tipo objeto (onde podem haver representações de nulo como strings)
    colunas_objeto = df.select_dtypes(include='object').columns

    # Lista de valores comuns que podem representar nulos em colunas de string
    valores_nulos_comuns = [' ', '', 'None', 'NULL', 'N/A', '-', '?']

    # Substitui esses valores por NaN nas colunas de objeto
    df[colunas_objeto] = df[colunas_objeto].replace(valores_nulos_comuns, np.nan)

    return df.isnull().sum(axis=1)


# Conta o número de nulos por linha (agora considerando as representações comuns)
nulos_por_linha = contar_nulos_por_linha(df.copy()) # Use .copy() para evitar modificar o DataFrame original diretamente

# Filtra as linhas que possuem mais de 2 valores nulos
linhas_com_mais_de_dois_nulos = nulos_por_linha[nulos_por_linha > 1]

# Obtém a contagem dessas linhas
contagem_linhas = len(linhas_com_mais_de_dois_nulos)

print(f"Número de linhas com mais de 2 valores nulos em diferentes colunas: {contagem_linhas}")


if contagem_linhas > 0:
    print("\nÍndices das linhas com mais de 2 valores nulos:")
    print(linhas_com_mais_de_dois_nulos.index.tolist())






#Conta quais linhas apresentam mais de um valor nulo e valor outlier

import pandas as pd
import numpy as np

def detectar_outliers_iqr_indices(df, coluna):
    """Detecta os índices das linhas que contêm outliers usando o método IQR."""
    Q1 = df[coluna].quantile(0.25)
    Q3 = df[coluna].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    outliers_indices = df[(df[coluna] < limite_inferior) | (df[coluna] > limite_superior)].index
    return set(outliers_indices)

def contar_nulos_por_linha(df):
    """Conta o número de valores nulos em cada linha do DataFrame."""
    colunas_objeto = df.select_dtypes(include='object').columns
    valores_nulos_comuns = [' ', '', 'None', 'NULL', 'N/A', '-', '?']
    df[colunas_objeto] = df[colunas_objeto].replace(valores_nulos_comuns, np.nan)
    return df.isnull().sum(axis=1)

#  Identificar os índices de outliers por coluna numérica
outlier_indices_por_coluna = {}
colunas_numericas = df.select_dtypes(include=['number']).columns
for coluna in colunas_numericas:
    if df[coluna].nunique() > 10:
        outlier_indices = detectar_outliers_iqr_indices(df.copy(), coluna)
        if outlier_indices:
            outlier_indices_por_coluna[coluna] = outlier_indices

#  Contar o número de nulos por linha
nulos_por_linha = contar_nulos_por_linha(df.copy())

# Identificar linhas com mais de um valor nulo
linhas_com_mais_de_um_nulo_indices = set(nulos_por_linha[nulos_por_linha > 1].index)

#  Identificar todos os índices de outliers
todos_indices_de_outliers = set()
for indices in outlier_indices_por_coluna.values():
    todos_indices_de_outliers.update(indices)

#  Encontrar a interseção entre as linhas com mais de um nulo e as linhas com pelo menos um outlier
linhas_com_nulos_e_outliers = linhas_com_mais_de_um_nulo_indices.intersection(todos_indices_de_outliers)

#  Contar o número dessas linhas
contagem_linhas = len(linhas_com_nulos_e_outliers)

print(f"Número de linhas com mais de um valor nulo (em diferentes colunas) E pelo menos um outlier (em diferentes colunas): {contagem_linhas}")

#  Mostrar os índices dessas linhas
if contagem_linhas > 0:
    print("\nÍndices das linhas com mais de um valor nulo E pelo menos um outlier:")
    print(sorted(list(linhas_com_nulos_e_outliers)))













#Mudança e tratamento inicial de dados
"""Remoção de linhas poluídas com muitos erros"""

# Lista dos índices das linhas que você quer remover (as 35 linhas com mais de 2 valores nulos)
indices_para_remover = [32, 101, 169, 250, 393, 403, 450, 487, 538, 548, 572, 762, 1000, 1022, 1084, 1256, 1392, 1431, 1450, 1491, 1619, 1697, 1837, 1883, 1887, 1908, 2037, 2315, 2455, 2889, 2921, 3052, 3227, 3247, 3263]

# Remove as linhas com os índices especificados
df_sem_linhas_nulos = df.drop(index=indices_para_remover)


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




# #Tratamento
# """Mudar colunas de float para int nos casos desnecessarios"""

# colunas_para_converter= ['x_maximo', 'soma_da_luminosidade','maximo_da_luminosidade', 'espessura_da_chapa_de_aço']
#     for coluna in colunas_para_converter:
#         if coluna in df.columns:
#             if df[coluna].dtype == 'float64':
#                 # Tenta converter para int, valores com parte decimal não zero
#                 # ou NaN serão convertidos para NaN no tipo Int64
#                 df[coluna] = df[coluna].astype('Int64')
#             else:
#                 print(f"A coluna '{coluna}' não é do tipo float64 e não será convertida.")
#         else:
#             print(f"A coluna '{coluna}' não existe no DataFrame.")






#Analisar o comportamento e ação depois de aplicar esse metodo para tratar outliers
#Apenas caps
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Função para aplicar capping (IQR)
def cap_iqr(series, k=1.5):
    Q1, Q3 = series.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - k * IQR, Q3 + k * IQR
    return series.clip(lower=lower, upper=upper)

# Aplicando capping
df_capped = df.copy()
for col in df_capped.select_dtypes(include=[np.number]).columns:
    df[col] = cap_iqr(df_capped[col])

# 5) Verificação final de outliers
def count_outliers_iqr(series, k=1.5):
    Q1, Q3 = series.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - k*IQR, Q3 + k*IQR
    return ((series < lower) | (series > upper)).sum()

print("Outliers após capping único nos limites originais:")
for col in numeric_cols:
    n = count_outliers_iqr(df_capped[col])
    print(f"{col:30s}: {n} outliers")
    
    
#Apenas winsorize
##Analisar o comportamento e ação depois de aplicar esse metodo para tratar outliers
from scipy.stats.mstats import winsorize

# Aplicando Winsorize (5% caudas)
df_winsorized = df.copy()
for col in df_winsorized.select_dtypes(include=[np.number]).columns:
    arr = winsorize(df_winsorized[col].values, limits=[0.05, 0.05])
    df_winsorized[col] = arr.data.astype(float)

# 5) Verificação final de outliers
def count_outliers_iqr(series, k=1.5):
    Q1, Q3 = series.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - k*IQR, Q3 + k*IQR
    return ((series < lower) | (series > upper)).sum()

print("Outliers após capping único nos limites originais:")
for col in numeric_cols:
    n = count_outliers_iqr(df_winsorized[col])
    print(f"{col:30s}: {n} outliers") 

#Apenas robust
##Analisar o comportamento e ação depois de aplicar esse metodo para tratar outliers
from sklearn.preprocessing import RobustScaler

# Aplicando RobustScaler
df_robust = df.copy()
rs = RobustScaler()
numeric_cols = df_robust.select_dtypes(include=[np.number]).columns
df_robust[numeric_cols] = rs.fit_transform(df_robust[numeric_cols])

# 5) Verificação final de outliers
def count_outliers_iqr(series, k=1.5):
    Q1, Q3 = series.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - k*IQR, Q3 + k*IQR
    return ((series < lower) | (series > upper)).sum()

print("Outliers após capping único nos limites originais:")
for col in numeric_cols:
    n = count_outliers_iqr(df_robust[col])
    print(f"{col:30s}: {n} outliers")


# Função para imputar a mediana nos outliers
##Analisar o comportamento e ação depois de aplicar esse metodo para tratar outliers
def median_imputation(series):
    median = series.median()
    Q1, Q3 = series.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    series.loc[(series < lower) | (series > upper)] = median
    return series

# Aplicando imputação pela mediana
df_median = df.copy()
for col in df_median.select_dtypes(include=[np.number]).columns:
    df_median[col] = median_imputation(df_median[col])

def count_outliers_iqr(series, k=1.5):
    Q1, Q3 = series.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - k*IQR, Q3 + k*IQR
    return ((series < lower) | (series > upper)).sum()

print("Outliers após capping único nos limites originais:")
for col in numeric_cols:
    n = count_outliers_iqr(df_median[col])
    print(f"{col:30s}: {n} outliers")
    
    
#Apenas remoção
#Analisar o comportamento e ação depois de aplicar esse metodo para tratar outliers
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    mask = (df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)
    return df[mask]

# Removendo outliers
df_removed = df.copy()
for col in df_removed.select_dtypes(include=[np.number]).columns:
    df_removed = remove_outliers_iqr(df_removed, col)
# 5) Verificação final de outliers
def count_outliers_iqr(series, k=1.5):
    Q1, Q3 = series.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - k*IQR, Q3 + k*IQR
    return ((series < lower) | (series > upper)).sum()

print("Outliers após capping único nos limites originais:")
for col in numeric_cols:
    n = count_outliers_iqr(df_removed[col])
    print(f"{col:30s}: {n} outliers")
    

#Grafico para comparar e entender os resultados e efeitos de usar esses metododos nos dados
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt

# Função para contar outliers pelo método IQR
def count_outliers_iqr(series, k=1.5):
    Q1, Q3 = series.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - k * IQR, Q3 + k * IQR
    return ((series < lower) | (series > upper)).sum()

# Lista de colunas numéricas para tratamento
num_cols = df.select_dtypes(include=[np.number]).columns

# Função para tratamento e contagem de outliers
def tratamento_outliers(df, metodo):
    df_tratado = df.copy()
    
    if metodo == 'winsorize':
        for col in num_cols:
            arr = winsorize(df_tratado[col].values, limits=[0.05, 0.05])
            df_tratado[col] = arr.data.astype(float)

    elif metodo == 'cap':
        for col in num_cols:
            Q1, Q3 = df_tratado[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            df_tratado[col] = df_tratado[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)

    elif metodo == 'mediana':
        for col in num_cols:
            mediana = df_tratado[col].median()
            df_tratado[col] = df_tratado[col].fillna(mediana)

    elif metodo == 'robust':
        rs = RobustScaler()
        df_tratado[num_cols] = rs.fit_transform(df_tratado[num_cols])

    # Contagem de outliers após tratamento
    outliers_count = {col: count_outliers_iqr(df_tratado[col]) for col in num_cols}
    return pd.Series(outliers_count, name=metodo)

# Aplicando os tratamentos
resultados = pd.DataFrame()
for metodo in ['winsorize', 'cap', 'mediana', 'robust']:
    outliers_count = tratamento_outliers(df, metodo)
    resultados = pd.concat([resultados, outliers_count], axis=1)

# Gráfico comparativo dos tratamentos
plt.figure(figsize=(10, 6))
resultados.plot(kind='bar', figsize=(14, 8), colormap='viridis')
plt.title('Comparação dos Tratamentos de Outliers')
plt.xlabel('Variáveis Numéricas')
plt.ylabel('Número de Outliers Restantes')
plt.xticks(rotation=45)
plt.legend(title='Método')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#Outra forma de conseguir comparar esses modelos e visualizar

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import RobustScaler

# Função para contar outliers pelo método IQR
def count_outliers_iqr(series, k=1.5):
    Q1, Q3 = series.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - k * IQR, Q3 + k * IQR
    return ((series < lower) | (series > upper)).sum()

# Função para Capping via IQR
def cap_iqr(series, k=1.5):
    Q1, Q3 = series.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    return series.clip(lower=Q1 - k * IQR, upper=Q3 + k * IQR)

# Função para Remover Outliers via IQR
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    mask = (df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)
    return df.loc[mask, column]

# Métodos de tratamento
def treat_outliers(df, method):
    treated_df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].nunique() > 10:
            if method == "winsorize":
                arr = winsorize(df[col].values, limits=[0.05, 0.05])
                treated_df[col] = arr.data.astype(float)
            elif method == "cap_iqr":
                treated_df[col] = cap_iqr(df[col])
            elif method == "robust":
                scaler = RobustScaler()
                treated_df[[col]] = scaler.fit_transform(df[[col]])
            elif method == "mediana":
                median = df[col].median()
                treated_df[col] = np.where((df[col] < df[col].quantile(0.05)) |
                                           (df[col] > df[col].quantile(0.95)), median, df[col])
            elif method == "remove_iqr":
                treated_df[col] = remove_outliers_iqr(df, col)
    return treated_df

# Aplicando os tratamentos e armazenando resultados
methods = ["winsorize", "cap_iqr", "robust", "mediana", "remove_iqr"]
outliers_count = {}

for method in methods:
    treated_df = treat_outliers(df, method)
    outlier_counts = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        if treated_df[col].nunique() > 10:
            n_out = count_outliers_iqr(treated_df[col])
            outlier_counts[col] = n_out
        else:
            outlier_counts[col] = 0  # Garantir que todas as colunas apareçam
    outliers_count[method] = outlier_counts

# Convertendo resultados para DataFrame e garantindo que todas as variáveis estejam presentes
outliers_df = pd.DataFrame(outliers_count).reindex(df.select_dtypes(include=[np.number]).columns)

# Gráfico comparativo - Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(outliers_df, annot=True, cmap="YlGnBu", fmt=".0f")
plt.title("Comparação de Métodos de Tratamento de Outliers")
plt.xlabel("Método")
plt.ylabel("Variável")
plt.show()

# Gráficos de Boxplot para cada método
for method in methods:
    treated_df = treat_outliers(df, method)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=treated_df.select_dtypes(include=[np.number]), orient="h")
    plt.title(f"Boxplot após Tratamento: {method}")
    plt.show()
                
    
















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






import pandas as pd
planilha_teste = pd.read_csv('bootcamp_test.csv')

# Armazenar a coluna 'id' para manter no resultado final
if 'id' in planilha_teste.columns:
    ids = planilha_teste['id']
    planilha_teste = planilha_teste.drop(columns=['id'])
else:
    # Caso não tenha a coluna 'id', criar IDs sequenciais
    print("Coluna 'id' não encontrada! Criando coluna ID com valores sequenciais...")
    ids = pd.Series(range(1, len(planilha_teste) + 1), name='id')

# Garantir que a planilha tenha as mesmas colunas do treino
colunas_treino = X.columns  # Colunas usadas no treinamento
for coluna in colunas_treino:
    if coluna not in planilha_teste.columns:
        planilha_teste[coluna] = 0

# Reordenar as colunas para garantir o mesmo formato
planilha_teste = planilha_teste[colunas_treino]

# Pré-processar os dados da planilha de teste
planilha_teste_scaled = xgb_clf.named_steps['preprocessing'].transform(planilha_teste)

# Converter novamente para DataFrame após a transformação
planilha_teste_scaled = pd.DataFrame(planilha_teste_scaled, columns=colunas_treino)

# Prever as probabilidades para cada linha da planilha de teste
probs = xgb_clf.predict_proba(planilha_teste_scaled)

# Ajustar o formato das probabilidades para um DataFrame
probabilidades = pd.DataFrame(probs, columns=label_encoder.classes_)

# Adicionar a coluna ID ao resultado final
resultado = pd.concat([ids.reset_index(drop=True), probabilidades], axis=1)

# Renomear as colunas para incluir 'falha_' antes dos números, usando o inverso do LabelEncoder
resultado.columns = ['ID'] + [f"falha_{label}" for label in label_encoder.inverse_transform(range(len(probabilidades.columns)))]

# Visualizar os resultados
print(resultado.head())

# Salvar o resultado em um arquivo CSV
resultado.to_csv("resultado_probabilidades.csv", index=False)
print("Arquivo 'resultado_probabilidades.csv' gerado com sucesso!")








#Treino para modelo com tratamento de outlier
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.preprocessing import RobustScaler, LabelEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestClassifier, VotingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from xgboost import XGBClassifier
# from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
# import matplotlib.pyplot as plt

# # 1) Gerar y_multiclass antes de tratar os outliers
# falhas = ['falha_1','falha_2','falha_3','falha_4','falha_5','falha_6','falha_outros']
# label_map = {name: idx for idx,name in enumerate(falhas)}
# df['y_multiclass'] = df[falhas].idxmax(axis=1).map(label_map)
# df['y_multiclass'] = LabelEncoder().fit_transform(df['y_multiclass'])

# # 2) Separar features e target
# feature_cols = [c for c in df.columns if c not in falhas + ['y_multiclass']]
# X = df[feature_cols].copy()
# y = df['y_multiclass'].copy()

# # 3) Função de capping apenas nas features
# def cap_iqr(series, k=1.5):
#     Q1, Q3 = series.quantile([0.25, 0.75])
#     IQR = Q3 - Q1
#     lower, upper = Q1 - k*IQR, Q3 + k*IQR
#     return series.clip(lower=lower, upper=upper)

# X_capped = X.copy()
# for col in X_capped.select_dtypes(include=[np.number]).columns:
#     X_capped[col] = cap_iqr(X_capped[col])

# # 4) Agora divida treino/teste usando X_capped
# X_train, X_test, y_train, y_test = train_test_split(
#     X_capped, y, test_size=0.2, random_state=42, stratify=y
# )

# # 5) Pré-processamento com RobustScaler (ou outro de sua escolha)
# numerical = X_capped.select_dtypes(include=[np.number]).columns
# preprocessor = ColumnTransformer([('num', RobustScaler(), numerical)])

# # 6) Definição dos pipelines de modelos
# models = {
#     'RandomForest': Pipeline([('prep',preprocessor),
#                               ('clf',RandomForestClassifier(random_state=42,class_weight='balanced'))]),
#     'XGBoost':      Pipeline([('prep',preprocessor),
#                               ('clf',XGBClassifier(objective='multi:softprob',
#                                                    eval_metric='mlogloss',random_state=42))]),
#     'SVM':          Pipeline([('prep',preprocessor),
#                               ('clf',SVC(probability=True,random_state=42))]),
#     'KNN':          Pipeline([('prep',preprocessor),
#                               ('clf',KNeighborsClassifier())]),
#     'Logistic':     Pipeline([('prep',preprocessor),
#                               ('clf',LogisticRegression(max_iter=1000,
#                                                         class_weight='balanced',
#                                                         random_state=42))]),
#     'DecisionTree': Pipeline([('prep',preprocessor),
#                               ('clf',DecisionTreeClassifier(random_state=42,
#                                                             class_weight='balanced'))])
# }

# # 7) Ensemble
# ensemble = VotingClassifier(
#     estimators=[(n,pipe) for n,pipe in models.items() if n in ('RandomForest','XGBoost','SVM')],
#     voting='soft'
# )
# models['Ensemble'] = ensemble

# # ... (suas importações e preparações anteriores)

# # Depois de ter X_train, X_test, y_train, y_test e o dicionário models

# class_names = list(label_map.keys())  # ['falha_1', 'falha_2', ..., 'falha_outros']

# for name, model in models.items():
#     print(f"\n*** {name} ***")
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)

#     # Assegura que y_test e y_pred são vetores 1D
#     y_true = np.asarray(y_test).ravel()
#     y_hat  = np.asarray(y_pred).ravel()

#     # Relatório de classificação
#     print(classification_report(
#         y_true,
#         y_hat,
#         target_names=class_names,
#         zero_division=0
#     ))

#     # Matriz de confusão
#     cm = confusion_matrix(y_true, y_hat)
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
#     disp.plot(cmap='Blues')
#     plt.title(f"Matriz de Confusão – {name}")
#     plt.show()

#     # ROC AUC macro (se disponível)
#     if hasattr(model, "predict_proba"):
#         y_proba = model.predict_proba(X_test)
#         y_bin   = pd.get_dummies(y_true)
#         auc = roc_auc_score(y_bin, y_proba, average='macro', multi_class='ovr')
#         print(f"ROC AUC (macro): {auc:.4f}")

#     # Validação cruzada
#     cv = cross_val_score(model, X, y, cv=5, scoring='accuracy')
#     print(f"CV accuracy: {cv}\nMédia: {cv.mean():.4f} ± {cv.std():.4f}")
