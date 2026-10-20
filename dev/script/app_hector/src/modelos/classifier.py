import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)


class Classifier:
    """
    Classificador genérico que encapsula diferentes modelos de
    aprendizado de máquina para visualização no Streamlit.
        Parâmetros
        ----------
        df : pd.DataFrame
            DataFrame contendo as colunas 'Dataset' e 'Classe_Predita'.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

        return None

    def testar_hipotese(self) -> None:
        """
        Testa a hipótese de que a distribuição das classes no treino e no
        teste é estatisticamente igual (homogênea)
        ou diferente (heterogênea) usando o teste Qui-Quadrado.

        Retorna:
            None
        """
        # 1. Criando a Tabela de Contingência (Frequências Observadas)
        tabela_contingencia = pd.crosstab(
            self.df["Dataset"],
            self.df["Classe_Predita"],
        )
        st.write("Tabela de Contingência (Observada):")
        st.dataframe(tabela_contingencia.T)
        st.write("-" * 40)

        # 2. Realizando o Teste Qui-Quadrado
        chi2, p_valor, graus, _ = chi2_contingency(tabela_contingencia)

        # 3. Exibindo os resultados
        st.write(f"Estatística Qui-Quadrado: {chi2:.4f}")
        st.write(f"Graus de Liberdade: {graus}")
        st.write(f"Valor-p: {p_valor:.4f}")

        # 4. Tomada de Decisão (usando nível de significância de 5%)
        alpha = 0.05
        st.write("\nConclusão:")
        if p_valor < alpha:
            mensagem = "Rejeitamos H0: "
            mensagem += "A distribuição das classes no treino e no teste é "
            mensagem += "<b style='color: red;'>ESTATISTICAMENTE DIFERENTE "
            mensagem += "(Heterogênea).</b>"
            st.markdown(mensagem, unsafe_allow_html=True)
        else:
            mensagem = "Falhamos em rejeitar H0: "
            mensagem += "A distribuição das classes no treino e no teste é "
            mensagem += "<b style='color: green;'>ESTATISTICAMENTE IGUAL "
            mensagem += "(Homogênea).</b>"
            st.markdown(mensagem, unsafe_allow_html=True)

        return None

    def matrix_confusao(self, df: pd.DataFrame, obs: str, pred: str) -> None:
        """
        Exibe a matriz de confusão e o relatório de classificação do modelo.
            Parâmetros
            ----------
            df : pd.DataFrame
                DataFrame com as classes observadas e preditas.
            obs : str
                Nome da coluna com os valores reais ('classe').
            pred : str
                Nome da coluna com as predições do modelo.
        """
        y_obs = df[obs]
        y_pred = df[pred]

        accuracy = accuracy_score(y_obs, y_pred)
        cm = confusion_matrix(y_obs, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=y_obs.value_counts().index,
        )
        disp.plot(cmap=plt.cm.Greens)
        plt.title(f"Accuracy {accuracy:.2f}")
        st.pyplot(plt)
        st.write("\nClassification Report:")
        st.dataframe(
            (pd.DataFrame(classification_report(y_obs, y_pred, output_dict=True)).T)
        )

        return None
