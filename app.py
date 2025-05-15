# app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
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

st.set_page_config(page_title="Dashboard Chapas de Aço", layout="wide")

# Carregamento com cache
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

df = load_data("bootcamp_train.csv")
df2= df.copy



# Navegação
section = st.sidebar.selectbox(
    "Seção",
    ["EDA", "Modelagem"]
)

if section == "EDA":

    st.header("Análise Exploratória de Dados")

    # 1) Normalizar as colunas de falha
    mapa_falha = {
        'False':0,'0':0,'Não':0,'nao':0,'N':0,'n':0,False:0,
        'True':1,'1':1,'Sim':1,'sim':1,'S':1,'y':1,True:1
    }
    falhas = [f"falha_{i}" for i in range(1,7)] + ["falha_outros"]
    df_eda = df
    for col in falhas:
        df_eda[col] = df_eda[col].astype(str).str.strip().map(mapa_falha).fillna(0).astype(int)

    # Pie: distribuição de falhas
    freq = df_eda[falhas].sum().sort_values(ascending=False)
    fig1, ax1 = plt.subplots(figsize=(6,6))
    ax1.pie(freq, labels=freq.index, autopct='%1.1f%%', startangle=90,
            colors=['gold','lightcoral','lightskyblue','lightgreen','orange','purple','pink'])
    ax1.axis('equal')
    st.subheader("Distribuição dos Tipos de Falha")
    st.pyplot(fig1)

    # 2) Mapeamento aço A300/A400
    mapping_a300 = {'não':0,'nao':0,'n':0,'0':0,'-':np.nan,
                    'sim':1,'s':1,'1':1,'yes':1,'y':1,'Sim':1,'Não':0,'N':0}
    mapping_a400 = {'não':0,'nao':0,'n':0,'0':0,'nan':np.nan,
                    'sim':1,'s':1,'1':1,'yes':1,'y':1,'Sim':1,'Não':0,'S':1}
    df_eda['tipo_do_aço_A300'] = df_eda['tipo_do_aço_A300'].str.lower().map(mapping_a300).astype(float)
    df_eda['tipo_do_aço_A400'] = df_eda['tipo_do_aço_A400'].str.lower().map(mapping_a400).astype(float)
    st.subheader("Valores nulos após mapeamento A300/A400")
    st.write(df_eda[['tipo_do_aço_A300','tipo_do_aço_A400']].isnull().sum())

    # 3) Valores negativos por coluna
    num_cols = df_eda.select_dtypes(include=np.number).columns
    neg_counts = (df_eda[num_cols] < 0).sum()
    neg_cols = neg_counts[neg_counts>0]
    fig2, ax2 = plt.subplots(figsize=(10,4))
    neg_cols.plot.bar(ax=ax2, color='skyblue')
    ax2.set_title("Contagem de Valores Negativos")
    ax2.set_ylabel("Quantidade")
    st.pyplot(fig2)

    # 4) Outliers (IQR) por coluna
    def count_iqr(s):
        Q1,Q3 = s.quantile([.25,.75])
        IQR = Q3-Q1
        return ((s< Q1-1.5*IQR)|(s>Q3+1.5*IQR)).sum()
    out_counts = df_eda[num_cols].apply(count_iqr)
    out_cols = out_counts[out_counts>0]
    fig3, ax3 = plt.subplots(figsize=(10,4))
    out_cols.plot.bar(ax=ax3, color='coral')
    ax3.set_title("Contagem de Outliers (IQR)")
    ax3.set_ylabel("Quantidade")
    st.pyplot(fig3)

    # 5) Heatmap de correlação
    st.subheader("Heatmap de Correlação")
    corr = df_eda.corr()
    fig4, ax4 = plt.subplots(figsize=(10,8))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, ax=ax4)
    st.pyplot(fig4)

    # 6) Boxplot de uma variável selecionada
    st.subheader("Boxplot por Falha")
    var = st.selectbox("Selecione variável:", num_cols)
    fig5, ax5 = plt.subplots(figsize=(6,4))
    sns.boxplot(data=df_eda, x='falha_outros', y=var, ax=ax5)  # exemplo por falha_outros
    st.pyplot(fig5)

    # 7) Barras de valores nulos
    st.subheader("Distribuição de Valores Nulos")
    nulls = df_eda.isnull().sum()
    nulls = nulls[nulls>0]
    fig6, ax6 = plt.subplots(figsize=(10,4))
    nulls.plot.bar(ax=ax6, color='mediumseagreen')
    ax6.set_ylabel("Quantidade de Nulos")
    st.pyplot(fig6)

    
     



elif section == "Modelagem":
    st.header("Modelagem e Avaliação Multiclasse")

    # --- 1) Clone do df de EDA para esta seção ---
    df2 = df.copy()

    # --- 2) Remoção das linhas com muitos nulos ---
    indices_para_remover = [
        32, 101, 169, 250, 393, 403, 450, 487, 538, 548, 572, 762, 1000, 1022,
        1084, 1256, 1392, 1431, 1450, 1491, 1619, 1697, 1837, 1883, 1887, 1908,
        2037, 2315, 2455, 2889, 2921, 3052, 3227, 3247, 3263
    ]
    df2.drop(index=indices_para_remover, errors="ignore", inplace=True)

    # --- 3) Normalização das colunas de falha para 0/1 ---
    mapa_falha = {
        'False':0,'0':0,'Não':0,'nao':0,'N':0,'n':0, False:0,
        'True':1,'1':1,'Sim':1,'sim':1,'S':1,'y':1, True:1
    }
    falhas = [f"falha_{i}" for i in range(1,7)] + ["falha_outros"]
    for col in falhas:
        df2[col] = (df2[col]
            .astype(str).str.strip()
            .map(mapa_falha)
            .fillna(0).astype(int)
        )

    # --- 4) Binárias A300/A400: mapear strings, imputar moda ---
    import numpy as np
    map_a300 = {'sim':1,'s':1,'1':1,'yes':1,'y':1,'nao':0,'não':0,'n':0,'0':0,'-':np.nan}
    map_a400 = map_a300.copy()
    df2['tipo_do_aço_A300'] = (df2['tipo_do_aço_A300']
        .astype(str).str.lower().map(map_a300).astype(float)
    )
    df2['tipo_do_aço_A400'] = (df2['tipo_do_aço_A400']
        .astype(str).str.lower().map(map_a400).astype(float)
    )
    # Imputa moda
    df2['tipo_do_aço_A300'].fillna(df2['tipo_do_aço_A300'].mode()[0], inplace=True)
    df2['tipo_do_aço_A400'].fillna(df2['tipo_do_aço_A400'].mode()[0], inplace=True)

    # --- 5) Preenchimento de nulos numéricos pela mediana ---
    num_na = [
        'x_maximo','soma_da_luminosidade','maximo_da_luminosidade',
        'espessura_da_chapa_de_aço','index_quadrado',
        'indice_global_externo','indice_de_luminosidade'
    ]
    for c in num_na:
        df2[c] = pd.to_numeric(df2[c], errors="coerce")
        df2[c].fillna(df2[c].median(), inplace=True)

    # --- 6) Drop de colunas irrelevantes e correção de sinais negativos ---
    df2.drop(columns=['id','peso_da_placa'], inplace=True)
    for c in df2.select_dtypes(include="number").columns:
        df2[c] = df2[c].abs()

    # --- 7) Criação do target multiclass ---
    label_map = {name: idx for idx, name in enumerate(falhas)}
    df2['y_multiclass'] = (
        df2[falhas]
          .idxmax(axis=1)
          .map(label_map)
    )
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df2['y_multiclass'] = le.fit_transform(df2['y_multiclass'])

    # --- 8) Split e pipeline de modelagem ---
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score

    X = df2.drop(columns=falhas+['y_multiclass'])
    y = df2['y_multiclass']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    numeric_feats = X.select_dtypes(include="number").columns
    preproc = ColumnTransformer([("num", StandardScaler(), numeric_feats)])

    # define pipelines
    models = {
      "RF": Pipeline([("prep",preproc),("clf",RandomForestClassifier(random_state=42,class_weight="balanced"))]),
      "XGB": Pipeline([("prep",preproc),("clf",XGBClassifier(objective="multi:softprob",
                                                            eval_metric="mlogloss",
                                                            random_state=42))]),
      "SVM": Pipeline([("prep",preproc),("clf",SVC(probability=True,random_state=42))]),
      "KNN": Pipeline([("prep",preproc),("clf",KNeighborsClassifier())]),
      "LR": Pipeline([("prep",preproc),("clf",LogisticRegression(max_iter=1000,
                                                                 class_weight="balanced",
                                                                 random_state=42))]),
      "DT": Pipeline([("prep",preproc),("clf",DecisionTreeClassifier(random_state=42,
                                                                     class_weight="balanced"))])
    }
    # ensemble suave
    from sklearn.ensemble import VotingClassifier
    models["ENS"] = VotingClassifier(
        estimators=[(n,pipe) for n,pipe in models.items() if n in ("RF","XGB","SVM")],
        voting="soft"
    )

    # --- 9) Treino, relatório e plot no Streamlit ---
    for name, pipe in models.items():
        st.subheader(f"📈 Modelo: {name}")
        with st.spinner(f"Treinando {name}…"):
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

        # classification report
        rpt = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        rpt_df = pd.DataFrame(rpt).T.round(2)
        st.write("**Classification Report**")
        st.dataframe(rpt_df)

        # confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(3,3))
        disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        plt.title(f"{name} CM")
        st.pyplot(fig)

        # ROC-AUC macro
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X_test)
            y_bin = pd.get_dummies(y_test)
            auc = roc_auc_score(y_bin, proba,
                                average="macro", multi_class="ovr")
            st.write(f"**ROC AUC (macro):** {auc:.3f}")

        # CV accuracy
        cv = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
        st.write(f"**CV accuracy:** {cv.round(3)} | mean={cv.mean():.3f} ± {cv.std():.3f}")
        st.markdown("---")
