import numpy as np
import pandas as pd
import streamlit as st
from joblib import load

from src.acesso_data import acesso_data_banco
from src.modelos.clustering import Clustering
from src.modelos.classifier import Classifier
from src.modelos.regression import Regression


class Paginas:
    @st.cache_data
    def busca_dados_cluster(
        _self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Função de busca dos dados de treino, validação e teste para clustering.
        Retorna:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            DataFrames dos datasets de treino, validação e teste.
        """
        banco = acesso_data_banco()
        x_train = load(banco + "01_X_train")
        x_valid = load(banco + "01_X_valid")
        x_test = load(banco + "01_X_test")
        y_train = load(banco + "01_y_train")
        y_valid = load(banco + "01_y_valid")
        y_test = load(banco + "01_y_test")
        feature_names = load(banco + "01_feature_names")

        df_train = pd.DataFrame(x_train, columns=feature_names)
        df_train["classe"] = y_train.values

        df_valid = pd.DataFrame(x_valid, columns=feature_names)
        df_valid["classe"] = y_valid.values

        df_test = pd.DataFrame(x_test, columns=feature_names)
        df_test["classe"] = y_test.values

        return df_train, df_valid, df_test

    @st.cache_data
    def busca_dados_class(
        _self,
    ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Função de busca dos resultados de classificação no treino e teste.
        Retorna:
            tuple[dict[str, pd.DataFrame], pd.DataFrame]:
            Dicionário com DataFrames de distribuição por modelo e DataFrame
            de teste com as predições.
        """
        banco = acesso_data_banco()
        df_train = load(banco + "03_df_train_resultados_classif")
        df_test = load(banco + "03_df_test_resultados_classif")

        modelos = ["GBM", "k-NN", "NB", "NN", "RF", "SVM"]
        datasets: dict[str, pd.DataFrame] = {}
        for nome in modelos:
            datasets[nome] = pd.DataFrame(
                {
                    "Dataset": (
                        ["Treino"] * len(df_train) + ["Teste"] * len(df_test)
                    ),
                    "Classe_Predita": np.concatenate(
                        [df_train[nome].values, df_test[nome].values]
                    ),
                }
            )

        return datasets, df_test

    @st.cache_data
    def busca_dados_regr(_self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Função de busca dos resultados de regressão no treino e teste.
        Retorna:
            tuple[pd.DataFrame, pd.DataFrame]:
            DataFrames dos datasets de treino e teste com as predições.
        """
        banco = acesso_data_banco()
        colunas = [
            "PEV",
            "Multiple Linear Regression",
            "Suport Vector Machine",
            "Bayesian Ridge",
            "Neural Network",
        ]
        df_train = load(banco + "04_df_train_resultados_regression")
        df_train = df_train[colunas].copy()

        df_test = load(banco + "04_df_test_resultados_regression")
        df_test = df_test[colunas].copy()

        return df_train, df_test

    @st.cache_data
    def __clustering(
        _self,
        variavel_1: str,
        variavel_2: str,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
    ) -> tuple[dict, str, str, str]:
        """
        Função de teste dos modelos de clustering.
        Parametros:
            _self: Referência para a própria classe.
            variavel_1: Primeira variável para o clustering.
            variavel_2: Segunda variável para o clustering.
            df_train: DataFrame de treino.
            df_test: DataFrame de teste.
        Retorna:
            tuple[dict, str, str, str]: Resultados dos modelos e
                mensagens de conclusão.
        """
        # K-means ------------------------------------------------
        i = variavel_1
        j = variavel_2

        cluster = Clustering(c1=i, c2=j, df=df_train, modelo="kmeans")
        resultados = {}
        resultados["K-means"], msn_kmenans = cluster.testar_modelo(
            best_param={"n_clusters": 3, "init": "k-means++", "n_init": 10},
            df_test=df_test,
        )

        # DBSCAN -------------------------------------------------
        cluster = Clustering(c1=i, c2=j, df=df_train, modelo="dbscan")
        resultados["DBSCAN"], msn_dbscan = cluster.testar_modelo(
            best_param={"eps": 0.5, "min_samples": 3, "metric": "euclidean"},
            df_test=df_test,
        )

        # Agglomerative ------------------------------------------
        cluster = Clustering(c1=i, c2=j, df=df_train, modelo="agglomerative")
        resultados["Agg"], msn_agg = cluster.testar_modelo(
            best_param={
                "n_clusters": 3,
                "metric": "euclidean",
                "linkage": "ward",
            },
            df_test=df_test,
        )

        return resultados, msn_kmenans, msn_dbscan, msn_agg

    @st.cache_data
    def __classifier(_self) -> None:
        dados, df_test_class = _self.busca_dados_class()

        col = st.columns((5, 0.5, 5, 0.5, 5))
        dados_metricas = {
            0: "GBM",
            2: "k-NN",
            4: "NB",
        }
        for k, v in dados_metricas.items():
            with col[k]:
                st.write("=" * 69)
                st.markdown(
                    f"<h4 style='text-align: center; '>{v}</h4>",
                    unsafe_allow_html=True,
                )
                st.write("=" * 69)
                classificador = Classifier(df=dados[v])
                classificador.testar_hipotese()
                st.divider()
                classificador.matrix_confusao(
                    df=df_test_class, obs="classe", pred=v
                )

        st.divider()

        col = st.columns((5, 0.5, 5, 0.5, 5))
        dados_metricas = {0: "NN", 2: "RF", 4: "SVM"}
        for k, v in dados_metricas.items():
            with col[k]:
                st.write("=" * 69)
                st.markdown(
                    f"<h4 style='text-align: center; '>{v}</h4>",
                    unsafe_allow_html=True,
                )
                st.write("=" * 69)
                classificador = Classifier(df=dados[v])
                classificador.testar_hipotese()
                st.divider()
                classificador.matrix_confusao(
                    df=df_test_class, obs="classe", pred=v
                )
        return None

    @st.cache_data
    def __regressor(_self) -> None:
        df_train_regr, df_test_regr = _self.busca_dados_regr()
        modelos = {
            0: df_train_regr.columns[1],
            2: df_train_regr.columns[2],
            4: df_train_regr.columns[3],
            6: df_train_regr.columns[4],
        }
        col = st.columns((5, 0.5, 5, 0.5, 5, 0.5, 5))
        for k, v in modelos.items():
            with col[k]:
                regressor = Regression(
                    y_pred_treino=df_train_regr[v],
                    y_pred_teste=df_test_regr[v],
                )
                regressor.testar_hipotese(y_pred=v)
                st.divider()
                regressor.obter_grafico(
                    df_test=df_test_regr, y_true="PEV", y_pred=v
                )
        return None

    @st.cache_data
    def __datasets(_self) -> None:
        df_train, df_valid, df_test = _self.busca_dados_cluster()
        col = st.columns((2, 0.05, 2, 0.05, 2))
        with col[0]:
            st.markdown(
                "<h4 style='text-align: center; '>Dados de Treino</h4>",
                unsafe_allow_html=True,
            )
            st.dataframe(df_train)

        with col[2]:
            st.markdown(
                "<h4 style='text-align: center; '>Dados de Validação</h4>",
                unsafe_allow_html=True,
            )
            st.dataframe(df_valid)

        with col[4]:
            st.markdown(
                "<h4 style='text-align: center; '>Dados de Teste</h4>",
                unsafe_allow_html=True,
            )
            st.dataframe(df_test)

        col = st.columns((2, 10))
        with col[0]:
            metr = pd.DataFrame(
                {
                    "Métrica": ["Treino", "Validação", "Teste"],
                    "Tamanho": [len(df_train), len(df_valid), len(df_test)],
                }
            )
            st.write("Tamanho dos datasets:")
            st.dataframe(metr)
        return None

    def metricas(self) -> str:
        """
        Função de apresentação das abas de métricas no Streamlit.
        Parametros:
            self: Referência para a própria classe.
        Retorna:
            str: Mensagem de sucesso.
        """
        abas = st.tabs(
            [
                "Clustering",
                "Classificação",
                "Regressão",
                "Datasets",
            ]
        )
        with abas[0]:
            df_train, _, df_test = self.busca_dados_cluster()
            col = st.columns((2, 2, 0.05, 2, 0.05, 2))
            with col[0]:
                variavel_1 = st.radio(
                    "Primeira variável:", ["Grupo_1", "Grupo_2", "Grupo_3"]
                )
                variavel_2 = st.radio(
                    "Segunda variável:", ["Grupo_2", "Grupo_3", "Grupo_1"]
                )

            resultados, msn_kmenans, msn_dbscan, msn_agg = self.__clustering(
                variavel_1=variavel_1,
                variavel_2=variavel_2,
                df_train=df_train,
                df_test=df_test,
            )
            with col[1]:
                st.markdown(f"<p>{msn_kmenans}</p>", unsafe_allow_html=True)
                st.pyplot(fig=resultados["K-means"])

            with col[3]:
                st.markdown(f"<p>{msn_dbscan}</p>", unsafe_allow_html=True)
                st.pyplot(fig=resultados["DBSCAN"])

            with col[5]:
                st.markdown(f"<p>{msn_agg}</p>", unsafe_allow_html=True)
                st.pyplot(fig=resultados["Agg"])

        with abas[1]:
            self.__classifier()

        with abas[2]:
            self.__regressor()

        with abas[3]:
            self.__datasets()

        return "Métricas Calculadas"
