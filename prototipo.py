#!/usr/bin/env python
# coding: utf-8

# # 1.0 Imports

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.neighbors import NearestNeighbors
from statsmodels.formula.api import ols


# # 2.0 Session State

# In[6]:


if "resultado" not in st.session_state:
    st.session_state["resultado"] = pd.DataFrame()

if "psm" not in st.session_state:
    st.session_state["psm"] = pd.DataFrame()

if "dados_modelo" not in st.session_state:
    st.session_state["dados_modelo"] = None

if "psm_info" not in st.session_state:
    st.session_state["psm_info"] = None

if "fig_psm_before" not in st.session_state:
    st.session_state["fig_psm_before"] = None

if "fig_psm_after" not in st.session_state:
    st.session_state["fig_psm_after"] = None

if "smd_executado" not in st.session_state:
    st.session_state["smd_executado"] = None

if "df_intervencao" not in st.session_state:
    st.session_state["df_intervencao"] = pd.DataFrame()


# # 3.0 Visual

# In[3]:


st.set_page_config(page_title="Projeto EGC - Avaliação e Monitoramento de Políticas Públicas", layout="centered")
st.title("Monitoramento e Avaliação de Políticas Públicas")


# # 4.0 Upload data

# In[4]:


if "historico_resultados" not in st.session_state:
    st.session_state.historico_resultados = []

# Mensagem inicial
st.info("""
#### Orientações iniciais

Esta ferramenta utiliza métodos de inferência causal para avaliação dos resultados.

Antes de executar sua análise:

- Organize em arquivos separados os dados (covariáveis) dos grupos controle e intervenção.
- Organize os dados do desfecho antes e após a exposição.

Importante:

- Os arquivos devem estar nos formatos `.csv` ou `.xlsx`.
- Não é possível fornecer variáveis categóricas
""")

# Upload arquivo grupo tratamento

uploaded_tratamento = st.file_uploader(
    "📁 Faça upload do arquivo de dados (.xlsx ou .csv) do grupo intervenção:",
    type = ["xlsx", "csv"],
    accept_multiple_files = False,
    key = "tratamento"
)

# Upload arquivo grupo controle
uploaded_controle = st.file_uploader(
    "📁 Faça upload do arquivo de dados (.xlsx ou .csv) do grupo controle:",
    type = ["xlsx", "csv"],
    accept_multiple_files = False,
    key = "controle"
)

df_combined = pd.DataFrame()


# # 5.0 Processamento

# In[5]:


if uploaded_tratamento or uploaded_controle:

    df_tratamento = None
    df_controle = None

    # ---- TRATAMENTO ----
    if uploaded_tratamento:
        try:
            if uploaded_tratamento.name.endswith(".xlsx"):
                df_tratamento = pd.read_excel(uploaded_tratamento, engine="openpyxl")
            else:
                df_tratamento = pd.read_csv(uploaded_tratamento)

            st.write("Visualização prévia dos dados grupo intervenção:")
            st.dataframe(df_tratamento.head())

            # Cria campo grupo
            df_tratamento["tratamento"] = 1

        except Exception as e:
            st.error(f"❌ Erro ao ler o arquivo {uploaded_tratamento.name}: {e}")

    # ---- CONTROLE ----
    if uploaded_controle:
        try:
            if uploaded_controle.name.endswith(".xlsx"):
                df_controle = pd.read_excel(uploaded_controle, engine="openpyxl")
            else:
                df_controle = pd.read_csv(uploaded_controle)

            st.write("Visualização prévia dos dados grupo controle:")
            st.dataframe(df_controle.head())

            # Cria campo grupo
            df_controle["tratamento"] = 0

        except Exception as e:
            st.error(f"❌ Erro ao ler o arquivo {uploaded_controle.name}: {e}")

    # ---- VERIFICAR COLUNAS ----
    if df_tratamento is not None and df_controle is not None:

        col_trat = set(df_tratamento.columns) 
        col_cont = set(df_controle.columns)

        if col_trat != col_cont:

            st.error("❌ Os arquivos possuem campos diferentes e não podem ser combinados.")

        else:

            # ---- COMBINAR DATAFRAMES ----
            try:
                df_combined = pd.concat([df_tratamento, df_controle], ignore_index=True)

                st.success(
                    f"✅ Arquivos processados com sucesso. "
                    f"Controle: {len(df_combined[df_combined['tratamento']==0])} registros | "
                    f"Intervenção: {len(df_combined[df_combined['tratamento']==1])} registros"
                )

            except Exception as e:
                st.error(
                    f"❌ Erro ao combinar os arquivos: {e}. Verifique se todos os campos são consistentes."
                )
                df_combined = pd.DataFrame()

    else:
        st.info("ℹ️ É necessário carregar os dois arquivos (controle e intervenção).")

else:
    st.info("ℹ️ Faça upload dos arquivos de intervenção e controle para começar.")


# # 6.0 Modelagem

# In[6]:


if not df_combined.empty:

    colunas_disponiveis = [c for c in df_combined.columns.tolist() if c != "tratamento"]

    colunas_usar = st.multiselect(
        "🔍 Selecione os campos de entrada (preditores) para o treinamento:",
        options = colunas_disponiveis)

    coluna_alvo = "tratamento"

    # Função de treinamento
    def treinar_modelo(modelo_nome, modelo):

        if not colunas_usar:
            st.error("❌ Por favor, selecione pelo menos um preditor.")
            return

        dados = df_combined[colunas_usar + [coluna_alvo]].copy()
        dados.dropna(inplace = True)

        # Separar features e target
        X = dados[colunas_usar].copy()
        y = dados[coluna_alvo].copy().values.ravel()

        # Divisão treino/teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, stratify = y, random_state = 42)

        # Treinamento
        try:
            modelo.fit(X_train, y_train)
        except Exception as e:
            st.error(f"❌ Erro durante o treinamento do modelo {modelo_nome}: {e}")
            return

        # Predições
        y_pred = modelo.predict(X_test)

        # Métricas
        matriz = confusion_matrix(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        st.session_state["modelo_treinado"] = True

        st.session_state["dados_modelo"] = {
            "modelo": modelo_nome,
            "matriz": matriz,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy
        }
    
#        st.subheader(f"📊 Métricas desepenho do modelo: {modelo_nome}")

#        col1, col2, col3, col4 = st.columns(4)

#        col1.metric("Precisão", f"{precision:.2f}")
#        col2.metric("Acurácia", f"{accuracy:.2f}")
#        col3.metric("Recall", f"{recall:.2f}")
#        col4.metric("F1-score", f"{f1:.2f}")

#        st.write("Matriz de Confusão:")
#        st.dataframe(matriz)

        # Realizar a predição da probabilidade de pertencer ao grupo tratamento e controle
        pred_proba = modelo.predict_proba(X)
        resultado = dados.copy()
        resultado['ps_1'] = pred_proba[:,1]
        resultado['ps'] = np.log(resultado['ps_1']/(1-resultado['ps_1']))
        
        
        st.session_state["resultado"] = resultado
        st.success("✅ Propensity Score calculado!")
    
    # -------------------------
    # Seleção do modelo
    # -------------------------

    st.subheader("🤖 Selecione o Modelo")
    
    if st.button("Regressão Logística"):
        lr = LogisticRegression(solver = "newton-cholesky", random_state = 42)
        treinar_modelo("Regressão Logística", lr)

# Adicionado novo para mostrar os resultados na tela sem que eles "sumam" ao executar a próxima ação
    if st.session_state["dados_modelo"] is not None:
    
        dados_modelo = st.session_state["dados_modelo"]
    
        st.subheader(
            f"📊 Métricas de desempenho do modelo: {dados_modelo['modelo']}"
        )
    
        col1, col2, col3, col4 = st.columns(4)
    
        col1.metric("Precisão", f"{dados_modelo['precision']:.2f}")
        col2.metric("Acurácia", f"{dados_modelo['accuracy']:.2f}")
        col3.metric("Recall", f"{dados_modelo['recall']:.2f}")
        col4.metric("F1-score", f"{dados_modelo['f1']:.2f}")
    
        st.write("Matriz de Confusão:")
        st.dataframe(dados_modelo["matriz"])


# # 7.0 PSM

# In[7]:


if not st.session_state["resultado"].empty:

    st.subheader("🔗 Pareamento (PSM)")
    
    # Função para realizar o PSM
    def psm(resultado):
        
        aux = resultado.copy()
        aux = aux.rename(columns={'tratamento':'Group'})
        nomes_novos = {
                        0: 'control',
                        1: 'intervention'
                      }
        aux['Group'] = aux['Group'].replace(nomes_novos)
        
        # Pareamento com método do vizinho mais próximo, 1:1, sem reposição, usando a distância máxima de 0,20 desvio padrão (capiler)
        caliper = np.std(resultado.ps) * 0.10
        n_neighbors = 10
        knn = NearestNeighbors(n_neighbors = n_neighbors, radius = caliper)
        ps = resultado[['ps']]
        knn.fit(ps)
        # Obtendo as distâncias e o index dos 10 vizinhos mais próximos de cada ponto
        distances, neighbor_indexes = knn.kneighbors(return_distance=True)
        # Para cada indivíduo do grupo tratamento, encontrar um indivíduo correspondente no grupo controle, sem repetição
        aux2 = resultado.copy().reset_index()
        matched_control = [] # Criar lista vazia para popular com os index do grupo controle pareados com um indivíduo do grupo tratamento
        for current_index, row in aux2.iterrows():  # Percorrer de forma iterativa todo o dataframe
            if row.tratamento == 0:  # A linha é correspondente ao grupo controle (não tratado)
                aux2.loc[current_index, 'matched'] = np.nan  # Criar uma coluna 'matched' e popular como NaN (dado não pareado)
            else:
                for idx in neighbor_indexes[current_index, :]: # Para cada linha "tratamento" encontrar os 10 vizinhos mais próximos
                    if aux2.loc[idx].tratamento == 0: 
                        if idx not in matched_control: # Controle para garantir que seja sem repetição
                            aux2.loc[current_index, 'matched'] = idx # Popular com o index
                            matched_control.append(idx) # Salvar o index já pareado no controle para evitar repetição
                            break
        
        treatment_matched = aux2.dropna(subset = ['matched'])  # Excluir exemplos sem pareamento
        control_matched_idx = treatment_matched.matched  # Id's dos municípios do grupo controle pareados
        control_matched_idx = control_matched_idx.astype(int)  
        control_matched = aux2.loc[control_matched_idx, :] 
        df_matched = pd.concat([treatment_matched, control_matched])
        df_aux = df_matched.copy()
        df_aux = df_aux.rename(columns={'tratamento':'Group'})
        nomes_novos = {
                        0: 'control',
                        1: 'intervention'
                        }
        df_aux['Group'] = df_aux['Group'].replace(nomes_novos)

#        st.write('Total de observações no grupo tratamento:', len(aux2[aux2.tratamento == 1]))
#        st.write('Total de observações no grupo tratamento pareados:', len(matched_control))
        
#        col1, col2 = st.columns(2)
#        fig1, ax1 = plt.subplots()
#        sns.kdeplot(data=aux, x='ps', hue='Group', fill=True, ax=ax1)
#        ax1.set_title("Distribuição de probabilidade antes do pareamento")
#        col1.pyplot(fig1)
        
#        fig2, ax2 = plt.subplots()
#        sns.kdeplot(data=df_aux, x='ps', hue='Group', fill=True, ax=ax2)
#        ax2.set_title("Distribuição após o pareamento")
#        col2.pyplot(fig2)

        st.session_state["psm"] = df_matched
        
        st.session_state["psm"] = df_matched
        st.session_state["psm_executado"] = True

        st.session_state["psm_total_tratamento"] = len(aux2[aux2.tratamento == 1])
        st.session_state["psm_total_pareados"] = len(matched_control)

        st.session_state["psm_aux"] = aux
        st.session_state["psm_df_aux"] = df_aux

        st.session_state["psm_executado"] = True

#        st.success("✅ PSM executado com sucesso!")

            
    # Executar pareamento

    st.subheader("Realizar pareamento")
    
    if st.button("Executar PSM"):
        psm(st.session_state["resultado"])


# Novo    
    if st.session_state.get("psm_executado", False):

        st.write(
            'Total de observações no grupo tratamento:',
            st.session_state["psm_total_tratamento"]
        )
    
        st.write(
            'Total de observações no grupo tratamento pareados:',
            st.session_state["psm_total_pareados"]
        )
    
        col1, col2 = st.columns(2)
    
        fig1, ax1 = plt.subplots()
        sns.kdeplot(
            data=st.session_state["psm_aux"],
            x='ps',
            hue='Group',
            fill=True,
            ax=ax1
        )
        ax1.set_title("Distribuição de probabilidade antes do pareamento")
        col1.pyplot(fig1)
    
        fig2, ax2 = plt.subplots()
        sns.kdeplot(
            data=st.session_state["psm_df_aux"],
            x='ps',
            hue='Group',
            fill=True,
            ax=ax2
        )
        ax2.set_title("Distribuição após o pareamento")
        col2.pyplot(fig2)
    
        st.success("✅ PSM executado com sucesso!")


# # 8.0 Análise estatística - SMD

# In[ ]:


if not st.session_state["psm"].empty:

    st.subheader("Verificar o balancemaneto das covariáveis")

    covariaveis = colunas_usar

    #covariaveis = [c for c in st.session_state["psm"].columns.tolist() if c != "tratamento"]

    #covariaveis = st.multiselect(
    #    "🔍 Selecione as covariáveis para executar análise estatística:",
    #    options = colunas_disponiveis)  

    # Função para análise SMD

    def teste_SMD(df, covariates):

        #if not covariaveis:
        #    st.error("❌ Selecione pelo menos uma covariável")
        #    return
            
        smd_values = {}
        for column in covariaveis:
            mean_treatment = df[column][df['tratamento'] == 1].mean()
            mean_control = df[column][df['tratamento'] == 0].mean()
            std_treatment = df[column][df['tratamento'] == 1].std()
            std_control = df[column][df['tratamento'] == 0].std()
            pooled_std = np.sqrt((std_treatment**2 + std_control**2) / 2)
            smd = abs(mean_treatment - mean_control) / pooled_std if pooled_std != 0 else 0
            smd_values[column] = smd
            smd_resultado = pd.DataFrame(list(smd_values.items()), columns=['Covariate', 'SMD'])
        return smd_resultado

     # Executar análise

    if st.button("Realizar análise estatística"):

        teste_smd_antes = teste_SMD(st.session_state["resultado"], covariaveis)
        teste_smd_depois = teste_SMD(st.session_state["psm"], covariaveis)
        
        aux3 = pd.merge(teste_smd_antes, teste_smd_depois, on = 'Covariate', how = 'left')
        aux4 = aux3.rename(columns={'SMD_x':'Unmatched', 'SMD_y':'Matched'})

        # Plotar resultados
        data_long = pd.melt(aux4, id_vars='Covariate', 
                            value_vars=['Unmatched', 'Matched'],
                            var_name='Group', value_name='SMD')
        
        # Ordenando as variáveis
        data_long['Covariate'] = pd.Categorical(data_long['Covariate'],
                                               categories=aux4.sort_values('Unmatched', ascending=False)['Covariate'],
                                               ordered=True)

        st.session_state["smd_executado"] = True

        # Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=data_long, x = 'SMD', y = 'Covariate', hue = 'Group', s = 30)
        ax.axvline(0.1, color = 'black', linestyle = '--')  # Linha de referência
        ax.set_title('Standardized Mean Difference')
        ax.legend(title=None)
        ax.grid(True, axis='x', linestyle='--', alpha=0.6)
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        st.pyplot(fig)   


# # 9.0 Análise de impacto

# In[7]:


if st.session_state["smd_executado"]:

    st.subheader("Executar análise de impacto")

    # Upload arquivo com os dados do desfecho
    uploaded_desfecho = st.file_uploader(
        "📁 Faça upload do arquivo de dados (.xlsx ou .csv) do desfecho antes e após a intervenção:",
        type=["xlsx", "csv"],
        accept_multiple_files=False,
    )

    if uploaded_desfecho:

        try:

            if uploaded_desfecho.name.endswith(".xlsx"):
                df_intervencao = pd.read_excel(
                    uploaded_desfecho,
                    engine="openpyxl"
                )
            else:
                df_intervencao = pd.read_csv(uploaded_desfecho)

            st.write("Visualização prévia dos dados de desfecho:")
            st.dataframe(df_intervencao.head())

        except Exception as e:
            st.error(
                f"❌ Erro ao ler o arquivo {uploaded_desfecho.name}: {e}"
            )
      
        dados_psm = st.session_state["psm"][['IBGE', 'tratamento']].reset_index(drop = True)
        df_impacto = pd.merge(dados_psm, df_intervencao, on = 'IBGE', how = 'left')

        df_aux = df_impacto[['tratamento', '2023_2', '2023_3']]

        # group g: 0 control group (PA), 1 treatment group (NJ)
        # t: 0 before treatment (min wage raise), 1 after treatment
        # gt: interaction of g * t
        
        # data before the treatment
        df_before = df_aux[['2023_2', 'tratamento']]
        df_before['t'] = 0
        df_before.columns = ['yll_rate', 'g', 't']
        
        # data after the treatment
        df_after = df_aux[['2023_3', 'tratamento']]
        df_after['t'] = 1
        df_after.columns = ['yll_rate', 'g', 't']
        
        # data for regression
        df_reg = pd.concat([df_before, df_after])

        # create the interaction 
        df_reg['gt'] = df_reg.g * df_reg.t
        
        ols_model = ols('yll_rate ~ g + t + gt', data=df_reg).fit()
        st.session_state["ols_model"] = ols_model
        
        st.success("✅ Análise executada")

        # Resultados

        impacto = ols_model.params['gt']
        p_valor = ols_model.pvalues['gt']

        st.subheader("📈 Resultado da Análise de Impacto")
        
        if p_valor < 0.05:
            st.success(
                f"✅ Houve evidência estatística de impacto da intervenção "
                f"(coeficiente = {impacto:.2f}; p = {p_valor:.4f})."
            )

            st.write(
                f"O efeito estimado da intervenção foi de **{impacto:.2f} anos de vida perdidos por morte por mil habitantes no grupo intervenção em comparação ao grupo controle**."
            )
        
        else:
            st.warning(
                f"⚠️ Não há evidência estatística suficiente para afirmar que a intervenção gerou resultado"
            )


        if st.button("Mostrar análise estatística completa"):
            #st.text(st.session_state["ols_model"].summary().as_text())
            st.code(str(ols_model.summary()))




# In[ ]:


#

