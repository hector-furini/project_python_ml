import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ks_2samp, levene, mannwhitneyu, spearmanr, ttest_ind
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
import streamlit as st


class Regression:
    """
    Regressor genérico que encapsula diferentes modelos de
    aprendizado de máquina para visualização no Streamlit.
        Parâmetros
        ----------
        y_pred_treino : pd.Series
            Série com os valores preditos no conjunto de treino.
        y_pred_teste : pd.Series
            Série com os valores preditos no conjunto de teste.
    """

    def __init__(
        self, y_pred_treino: pd.Series, y_pred_teste: pd.Series
    ) -> None:
        self.y_pred_treino = y_pred_treino.dropna()
        self.y_pred_teste = y_pred_teste.dropna()

        return None

    def __interpretar(
        self,
        nome_teste: str,
        p_valor: float,
        stat: float,
        alfa: float = 0.05,
    ) -> None:
        if p_valor < alfa:
            color = "red"
            resultado = "DIFERENTES (Rejeita H0)"
        else:
            color = "green"
            resultado = "IGUAIS (Falha ao rejeitar H0)"

        msn_resul = f"<h6 style='text-align: center;' >{nome_teste}</h6>"
        msn_resul += f"<h8 style='color: {color};' >{resultado}</h8>"
        st.markdown(msn_resul, unsafe_allow_html=True)

        st.dataframe(
            pd.DataFrame(
                {
                    "Valores": [
                        round(stat, 4),
                        round(p_valor, 4),
                    ],
                    "index": [
                        "Estatística",
                        "p-valor",
                    ],
                }
            )
            .set_index("index")
            .astype(str)
        )

        return None

    def obter_grafico(
        self, df_test: pd.DataFrame, y_true: str, y_pred: str
    ) -> None:
        df_limpo = df_test[[y_true, y_pred]].dropna()
        r_cal, r_cal_p = spearmanr(df_limpo[y_true], df_limpo[y_pred])
        r2_cal = r2_score(df_limpo[y_true], df_limpo[y_pred])
        mae_cal = mean_absolute_error(df_limpo[y_true], df_limpo[y_pred])
        rmse_cal = mean_squared_error(df_limpo[y_true], df_limpo[y_pred])
        mape_cal = mean_absolute_percentage_error(
            df_limpo[y_true], df_limpo[y_pred]
        )

        mensagem = f"""
    R Spearman: {r_cal:.4f}
    R p-value: {r_cal_p:.4f}
    R2: {r2_cal:.4f}
    MAE: {mae_cal:.4f}
    RMSE: {rmse_cal:.4f}
    MAPE: {mape_cal:.4f}
        """

        p = sns.jointplot(
            x=y_true,
            y=y_pred,
            kind="reg",
            marginal_ticks=True,
            data=df_limpo,
        )
        p.fig.suptitle(
            f"Correlação para {y_true} com algoritmo {y_pred}", y=1.02
        )
        plt.text(
            0.05,
            0.99,
            mensagem,
            transform=p.ax_joint.transAxes,
            fontsize=12,
            ha="left",
            va="top",
        )
        plt.xlabel("Observado")
        plt.ylabel("Predito")
        st.pyplot(p.fig)
        plt.close(p.fig)

        return None

    def testar_hipotese(self, y_pred: str) -> None:
        """
        Testa a hipótese de que a distribuição das predições no treino
        e no teste é estatisticamente igual (homogênea) ou diferente
        (heterogênea) usando testes estatísticos apropriados.
            Parâmetros:
                y_pred: str - Nome do modelo avaliado.
            Retorna:
                None
        """
        # 1. Teste de Kolmogorov-Smirnov (para distribuições)
        ks_stat, p_v_ks = ks_2samp(self.y_pred_treino, self.y_pred_teste)

        # 2. Teste T de Welch (para médias, assumindo variâncias
        # potencialmente diferentes)
        t_stat, p_val_t = ttest_ind(
            self.y_pred_treino, self.y_pred_teste, equal_var=False
        )

        # 3. Teste U de Mann-Whitney (para medianas, não-paramétrico)
        u_stat, p_val_u = mannwhitneyu(
            self.y_pred_treino, self.y_pred_teste, alternative="two-sided"
        )

        # 4. Teste de Levene (para variâncias/dispersão)
        w_stat, p_val_levene = levene(self.y_pred_treino, self.y_pred_teste)

        # Função auxiliar para imprimir resultados
        st.markdown(
            f"<h4 style='text-align: center;' >{y_pred}</h4>",
            unsafe_allow_html=True,
        )
        self.__interpretar("Teste Kolmogorov-Smirnov", p_v_ks, ks_stat)
        self.__interpretar("Teste T (Médias)", p_val_t, t_stat)
        self.__interpretar("Mann-Whitney (Medianas)", p_val_u, u_stat)
        self.__interpretar("Levene (Variâncias)", p_val_levene, w_stat)

        return None
