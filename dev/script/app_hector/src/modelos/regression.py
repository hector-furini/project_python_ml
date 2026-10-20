import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import streamlit as st


class Regressor:
    """Classe para apresentar resultados de modelos de Regressão"""

    def __init__(self, dados: pd.DataFrame):
        self.__dados = dados

    def print_avaliacao(self, y_true: np.ndarray, y_pred: np.ndarray):
        p = sns.jointplot(
            x=y_true,
            y=y_pred,
            kind="reg",
            truncate=False,
            color="m",
            height=4,
        )
        p.ax_joint.set_xlabel("Valor Real (Score_Impacto_Clinico)")
        p.ax_joint.set_ylabel("Valor Predito")
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        col1, col2, col3, col4 = st.columns(4)
        col2.pyplot(fig=p.fig)
        col3.write(f"- R²: {r2:.3f}")
        col3.write(f"- MAE: {mae:.3f}")
        col3.write(f"- RMSE: {rmse:.3f}")
        plt.close(p.fig)

    @property
    def dados(self):
        return self.__dados
