import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_fscore_support
import streamlit as st


class Classifier:
    """Classe para apresentar resultados de modelos de Classificação"""

    def __init__(self, dados: pd.DataFrame):
        self.__dados = dados

    def print_avaliacao(self, y_true: np.ndarray, y_pred: np.ndarray):
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, ax=ax, cmap=plt.cm.Greens
        )
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=np.nan
        )
        disp.ax_.set_title(f"Accuracy {accuracy:.2f}")

        col1, col2, col3, col4 = st.columns(4)
        col2.pyplot(fig=fig)
        col3.write(f"- Precision: {precision:.2f}")
        col3.write(f"- Recall: {recall:.2f}")
        col3.write(f"- F1-Score: {f1:.2f}")
        plt.close(fig)

    @property
    def dados(self):
        return self.__dados
