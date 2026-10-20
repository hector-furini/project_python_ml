import os
import pickle

import pandas as pd
from PIL import Image
import streamlit as st


class Data:
    """Classe para gerenciar acesso aos dados do projeto"""

    def __init__(self):
        self.__dir_data = "./app_hector/data/"
        if not os.path.isdir(self.__dir_data):
            self.__dir_data = "./dev/script/app_hector/data/"
        if not os.path.isdir(self.__dir_data):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.__dir_data = os.path.join(base_dir, "data") + "/"

    @staticmethod
    @st.cache_data
    def busca_dados_cluster(dir_data: str) -> dict:
        dict_retorno = {
            "01_original": pd.read_excel(
                dir_data + "01_Follow_Up_Dissertacao.xlsx"
            ),
            "01_preparado": pd.read_csv(
                dir_data + "01_Follow_Up_Clustering_Prep.csv"
            ),
            "02_cluster": pd.read_csv(
                dir_data + "02_Follow_Up_Clustering_Res.csv"
            ),
        }
        return dict_retorno

    @staticmethod
    @st.cache_data
    def busca_dados_class(dir_data: str) -> dict:
        dict_retorno = {
            "03_dados_prep": pd.read_csv(
                dir_data + "03_Follow_Up_Classification_Prep.csv"
            ),
            "04_metricas_treino": pd.read_csv(
                dir_data + "04_metricas_classificacao.csv"
            ),
        }
        with open(dir_data + "test_resultados_classif", "rb") as fp:
            dict_retorno["04_test_resultados"] = pickle.load(fp)
        return dict_retorno

    @staticmethod
    @st.cache_data
    def busca_dados_regr(dir_data: str) -> dict:
        dict_retorno = {
            "04_dados_prep": pd.read_csv(
                dir_data + "04_Follow_Up_Regression_Prep.csv"
            ),
            "04_metricas_treino": pd.read_csv(
                dir_data + "04_metricas_regressao.csv"
            ),
        }
        with open(dir_data + "train_resultados_regr", "rb") as fp:
            dict_retorno["04_train_resultados"] = pickle.load(fp)
        with open(dir_data + "test_resultados_regr", "rb") as fp:
            dict_retorno["04_test_resultados"] = pickle.load(fp)
        return dict_retorno

    def logo_datalab(self):
        caminho = self.__dir_data + "img/DataLab_Logo_i.jpg"
        if os.path.isfile(caminho):
            return Image.open(caminho)
        return None

    def icone_pagina(self):
        caminho = self.__dir_data + "img/PythonML.ico"
        if os.path.isfile(caminho):
            return Image.open(caminho)
        return None

    @property
    def dir_data(self):
        return self.__dir_data

    @dir_data.setter
    def dir_data(self, novo_dir: str):
        if os.path.isdir(novo_dir):
            self.__dir_data = novo_dir
