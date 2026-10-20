import pandas as pd
from src.acesso_data import Data
from src.modelos.classifier import Classifier
from src.modelos.clustering import Clustering
from src.modelos.regression import Regressor
import streamlit as st


class Paginas:
    """Classe para gerenciar as páginas do Streamlit"""

    def __init__(self):
        self.__data = Data()
        icone = self.__data.icone_pagina()
        st.set_page_config(
            page_title="Projeto ML - Follow-Up",
            page_icon=icone if icone is not None else "📊",
            layout="wide",
        )
        self.__paginas = {
            "Introdução": "Introdução",
            "Clusterização": "Clusterização",
            "Classificação": "Classificação",
            "Regressão": "Regressão",
            "Datasets": "Datasets",
        }
        self.__modelos_cluster = {
            "K-means": "K-means",
            "Gaussian Mixture": "Gaussian Mixture",
            "Agglomerative": "Agglomerative",
            "Spectral": "Spectral",
            "Affinity Propagation": "Affinity Propagation",
        }
        self.__modelos_class = [
            "Dummy",
            "Support Vector Machine",
            "Decision Tree",
            "Random Forest",
            "Extra Trees",
            "AdaBoost",
            "Gradient Boosting",
            "Multi-Layer Perceptron",
            "K-Neighbors",
            "Logistic Regression",
            "Gaussian Process",
            "Gaussian Naive Bayes",
            "Quadratic Discriminant",
            "Linear Discriminant",
        ]
        self.__modelos_regr = [
            "Dummy",
            "Support Vector Machine",
            "Decision Tree",
            "Random Forest",
            "Extra Trees",
            "AdaBoost",
            "Gradient Boosting",
            "Multi-Layer Perceptron",
            "K-Neighbors",
            "Linear Regression",
            "Gaussian Process",
            "Bayesian Ridge",
        ]

    def __clustering(self) -> dict:
        dict_dados = self.__data.busca_dados_cluster(self.__data.dir_data)
        cluster = Clustering(dict_dados["02_cluster"])
        dict_retorno = {}
        for _, m in self.__modelos_cluster.items():
            dict_retorno[m] = cluster.testar_modelo(m)
        return dict_retorno

    def __mostrar_clusters_row1(self, resultados: dict):
        col11, col12, col13 = st.columns(3)
        with col11:
            st.markdown("### **K-means**")
            st.pyplot(fig=resultados["K-means"].fig)
            st.write(
                """
                - 3 clusters bem delimitados no espaço de componentes principais
                - Separação equilibrada dos perfis clínicos e citogenéticos
                """
            )
        with col12:
            st.markdown("### **Gaussian Mixture**")
            st.pyplot(fig=resultados["Gaussian Mixture"].fig)
            st.write(
                """
                - Agrupamento probabilístico com 2 componentes principais
                - Captura a distinção primária de sobrevida e evolução
                """
            )
        with col13:
            st.markdown("### **Agglomerative**")
            st.pyplot(fig=resultados["Agglomerative"].fig)
            st.write(
                """
                - Clusterização hierárquica (k=3)
                - Fronteiras consistentes com o agrupamento por K-means
                """
            )

    def __mostrar_clusters_row2(self, resultados: dict):
        col21, col22, _ = st.columns(3)
        with col21:
            st.markdown("### **Spectral**")
            st.pyplot(fig=resultados["Spectral"].fig)
            st.write(
                """
                - Baseado em grafos de similaridade (k=2)
                - Destaca os dois grupos clínicos predominantes
                """
            )
        with col22:
            st.markdown("### **Affinity Propagation**")
            st.pyplot(fig=resultados["Affinity Propagation"].fig)
            st.write(
                """
                - Determinação automática de exemplares (11 subgrupos)
                - Sensível a variações locais nos dados clínicos
                """
            )

    def __mostrar_clustering(self, resultados: dict):
        st.markdown("# **Comparação de Modelos de Clusterização**")
        st.write(
            "Agrupamento não supervisionado aplicado aos dados clínicos e "
            "moleculares da coorte (redução dimensional via PCA: comp1 x comp2)."
        )
        self.__mostrar_clusters_row1(resultados)
        st.divider()
        self.__mostrar_clusters_row2(resultados)

    def __classifier(self, modelo: str):
        dict_dados = self.__data.busca_dados_class(self.__data.dir_data)
        classif = Classifier(dict_dados["03_dados_prep"])
        st.markdown(f"# **Avaliação do Modelo: {modelo}**")
        st.write("Alvo da classificação: **Cluster (K-means)**")

        df_met = dict_dados["04_metricas_treino"]
        df_mod = df_met[df_met["Modelo"] == modelo]
        if not df_mod.empty:
            st.markdown("## **Métricas da Validação Cruzada (Treino)**")
            st.dataframe(df_mod, hide_index=True)

        st.markdown("## **Desempenho na Base de Teste**")
        y_true = dict_dados["04_test_resultados"]["Target"]
        y_pred = dict_dados["04_test_resultados"][modelo]
        classif.print_avaliacao(y_true, y_pred)

        st.divider()
        st.markdown("## **Resumo Comparativo de Todos os Classificadores**")
        st.dataframe(df_met, hide_index=True)

    def __regressor(self, modelo: str):
        dict_dados = self.__data.busca_dados_regr(self.__data.dir_data)
        regr = Regressor(dict_dados["04_dados_prep"])
        st.markdown(f"# **Avaliação do Modelo: {modelo}**")
        st.write("Alvo da regressão: **Score_Impacto_Clinico**")

        df_met = dict_dados["04_metricas_treino"]
        df_mod = df_met[df_met["Modelo"] == modelo]
        if not df_mod.empty:
            st.markdown("## **Métricas da Validação Cruzada (Treino)**")
            st.dataframe(df_mod, hide_index=True)

        st.markdown("## **Desempenho na Base de Treino**")
        y_true_tr = dict_dados["04_train_resultados"]["Target"]
        y_pred_tr = dict_dados["04_train_resultados"][modelo]
        regr.print_avaliacao(y_true_tr, y_pred_tr)

        st.markdown("## **Desempenho na Base de Teste**")
        y_true_te = dict_dados["04_test_resultados"]["Target"]
        y_pred_te = dict_dados["04_test_resultados"][modelo]
        regr.print_avaliacao(y_true_te, y_pred_te)

        st.divider()
        st.markdown("## **Resumo Comparativo de Todos os Regressores**")
        st.dataframe(df_met, hide_index=True)

    def __datasets(self):
        dict_cluster = self.__data.busca_dados_cluster(self.__data.dir_data)
        dict_class = self.__data.busca_dados_class(self.__data.dir_data)
        dict_regr = self.__data.busca_dados_regr(self.__data.dir_data)

        st.markdown("# **Exploração dos Datasets**")
        abas = st.tabs(
            [
                "1. Dados Originais",
                "2. Prep. Clusterização",
                "3. Resultados Clusters",
                "4. Prep. Classificação",
                "5. Prep. Regressão",
            ]
        )
        with abas[0]:
            st.subheader("Base Original - Follow-Up Dissertação")
            st.dataframe(dict_cluster["01_original"])
        with abas[1]:
            st.subheader("Dados Preparados para Clusterização (Normalizados)")
            st.dataframe(dict_cluster["01_preparado"])
        with abas[2]:
            st.subheader("Componentes Principais e Rótulos dos Clusters")
            st.dataframe(dict_cluster["02_cluster"])
        with abas[3]:
            st.subheader("Dados Preparados para Classificação")
            st.dataframe(dict_class["03_dados_prep"])
        with abas[4]:
            st.subheader("Dados Preparados para Regressão")
            st.dataframe(dict_regr["04_dados_prep"])

    def __intro(self):
        logo = self.__data.logo_datalab()
        if logo is not None:
            st.sidebar.image(logo)
        st.markdown("# **Análise de Machine Learning — Coorte Follow-Up**")
        st.markdown(
            """
            Aplicação desenvolvida em **Streamlit** para visualização dos resultados
            de **Clusterização**, **Classificação** e **Regressão** aplicados aos
            dados clínicos, citogenéticos e moleculares do estudo de Follow-Up.

            ---
            ### **Estrutura do Projeto**
            1. **Clusterização:** Identificação não supervisionada de perfis de
               pacientes utilizando redução de dimensionalidade (PCA) e múltiplos
               algoritmos de agrupamento (*K-means*, *Gaussian Mixture*,
               *Agglomerative*, *Spectral* e *Affinity Propagation*).
            2. **Classificação:** Predição supervisionada dos perfis identificados
               pelo *K-means* comparando 14 algoritmos com validação cruzada
               (*GridSearchCV*).
            3. **Regressão:** Modelagem preditiva do **Score de Impacto Clínico**
               comparando 12 algoritmos de regressão avaliados por $R^2$, MAE e RMSE.
            4. **Datasets:** Consulta interativa às tabelas originais, pré-processadas
               e resultados gerados ao longo do pipeline.
            """
        )

    def mostrar_pagina(self):
        pagina = st.sidebar.selectbox("Selecione a página", self.__paginas)
        if pagina == self.__paginas["Introdução"]:
            self.__intro()
        elif pagina == self.__paginas["Clusterização"]:
            resultados = self.__clustering()
            self.__mostrar_clustering(resultados)
        elif pagina == self.__paginas["Classificação"]:
            modelo = st.sidebar.selectbox(
                "Selecione o Modelo", self.__modelos_class
            )
            self.__classifier(modelo)
        elif pagina == self.__paginas["Regressão"]:
            modelo = st.sidebar.selectbox(
                "Selecione o Modelo", self.__modelos_regr
            )
            self.__regressor(modelo)
        elif pagina == self.__paginas["Datasets"]:
            self.__datasets()

    @property
    def data(self):
        return self.__data

    @property
    def paginas(self):
        return self.__paginas

    @property
    def modelos_cluster(self):
        return self.__modelos_cluster

    @property
    def modelos_class(self):
        return self.__modelos_class

    @property
    def modelos_regr(self):
        return self.__modelos_regr

    @property
    def dados_cluster(self) -> pd.DataFrame:
        return self.__data.busca_dados_cluster(self.__data.dir_data)[
            "02_cluster"
        ]
