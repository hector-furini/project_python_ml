# Métricas de avaliação de clustering
from sklearn.metrics import adjusted_rand_score

# Bibliotecas para clustering
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

# Testagem dos modelos
from scipy.stats import chisquare
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid

# Bibliotecas para o modelo de clustering
import numpy as np
import pandas as pd

# Geração de gráficos
import seaborn as sns


class Clustering:
    """
    Classe para realizar clustering em uma bivariada de um DataFrame.
    """

    def __init__(
        self, c1: str, c2: str, df: pd.DataFrame, modelo: str
    ) -> None:
        self.c1 = c1
        self.c2 = c2
        self.df = df.dropna(subset=[c1, c2]).copy()
        self.modelo = modelo

        return None

    def _preparar_par(
        self, df_origem: pd.DataFrame
    ) -> tuple[pd.DataFrame, str, str]:
        """
        Prepara o par de variáveis garantindo colunas distintas caso
        o usuário selecione a mesma variável duas vezes.
        """
        df_limpo = df_origem.dropna(subset=[self.c1, self.c2])
        if self.c1 == self.c2:
            col_y = f"{self.c2}_copia"
            df_par = pd.DataFrame(
                {
                    self.c1: df_limpo[self.c1].values,
                    col_y: df_limpo[self.c2].values,
                },
                index=df_limpo.index,
            )
            return df_par, self.c1, col_y
        return df_limpo[[self.c1, self.c2]].copy(), self.c1, self.c2

    def _calcular_qui_quadrado(
        self,
        observed_val_counts: pd.Series,
        expected_val_counts: pd.Series,
    ) -> tuple[float, float]:
        """
        Executa o teste Qui-Quadrado de Aderência protegendo contra
        frequências esperadas nulas ou cluster único.
        """
        mask = expected_val_counts > 0
        obs_valid = observed_val_counts[mask]
        exp_valid = expected_val_counts[mask]
        if len(exp_valid) < 2:
            return 0.0, 1.0
        chi2_stat, p_value = chisquare(f_obs=obs_valid, f_exp=exp_valid)
        return float(chi2_stat), float(p_value)

    def _formatar_resultado(
        self,
        titulo: str,
        labels_train: np.ndarray,
        expected_val_counts: pd.Series,
        observed_val_counts: pd.Series,
        chi2_stat: float,
        p_value: float,
        ari_score: float,
    ) -> str:
        """
        Formata o relatório HTML das métricas de consistência de clustering.
        """
        msn_resul = "=" * 50 + "<br>"
        msn_resul += f"<h6 style='text-align: center;'>{titulo}</h6>"
        msn_resul += "=" * 50 + "<br>"
        msn_resul += "Clusters identificados no Treino:       "
        msn_resul += f"{np.unique(labels_train)}<br><br>"
        msn_resul += "Frequências Esperadas na Validação (Treino): <br>"
        msn_resul += f"{expected_val_counts.values.round(2)}<br><br>"
        msn_resul += "Frequências Observadas na Validação: <br>"
        msn_resul += f"       {observed_val_counts.values}<br>"
        msn_resul += "-" * 50 + "<br>"
        msn_resul += f"Estatística Qui-Quadrado (χ²): {chi2_stat:.4f}<br>"
        msn_resul += f"p-valor:                        {p_value:.4f}<br>"
        msn_resul += "-" * 50 + "<br><br>"

        if p_value > 0.05:
            msn_resul += (
                "Conclusão 1: Não rejeitamos H0. A distribuição dos "
                "clusters na validação "
                "<b style='color: green;'>É IGUAL</b>"
                " à do treino (p > 0.05).<br>"
            )
        else:
            msn_resul += (
                "Conclusão 2: Rejeitamos H0. A distribuição dos "
                "clusters na validação "
                "<b style='color: red;'>É DIFERENTE</b>"
                " da do treino (p <= 0.05).<br>"
            )

        msn_resul += "<br>" + "-" * 50 + "<br><br>"

        if ari_score > 0.5:
            msn_resul += (
                "Conclusão: O modelo de clustering do Treino e o "
                "modelo da Validação "
                "<b style='color: green;'>SÃO SEMELHANTES</b>"
                f" (ARI = {ari_score:.4f}).<br><br>"
            )
        else:
            msn_resul += (
                "Conclusão: O modelo de clustering do Treino e o "
                "modelo da Validação "
                "<b style='color: red;'>NÃO SÃO SEMELHANTES</b>"
                f" (ARI = {ari_score:.4f}).<br><br>"
            )

        return msn_resul

    def grafico_scatter(
        self, hue: str | list | np.ndarray, df_grafico: pd.DataFrame
    ) -> sns.JointGrid:
        """
        Função para gerar um gráfico de dispersão (scatter plot) com base
        em duas variáveis e uma variável de agrupamento (hue).
        """
        df_plot, col_x, col_y = self._preparar_par(df_grafico)
        p = sns.jointplot(
            x=col_x,
            y=col_y,
            hue=hue,
            height=6,
            ratio=4,
            marginal_ticks=True,
            data=df_plot,
        )
        p.fig.suptitle("Correlação Variáveis Discretas", y=1.02)
        p.set_axis_labels(xlabel=self.c1, ylabel=self.c2)
        return p

    def __testar_kmeans(
        self, best_param: dict, df_test: pd.DataFrame
    ) -> tuple[sns.JointGrid, str]:
        """
        Testa a consistência dos clusters K-Means entre treino e
        validação/teste via Qui-Quadrado de Aderência e ARI.
        """
        params = dict(best_param)
        params.setdefault("random_state", 42)

        x_train, _, _ = self._preparar_par(self.df)
        x_val, _, _ = self._preparar_par(df_test)

        kmeans_train = KMeans(**params).fit(x_train)
        labels_train = kmeans_train.labels_
        labels_val_predicted = kmeans_train.predict(x_val)

        kmeans_val_direct = KMeans(**params).fit(x_val)
        labels_val_direct = kmeans_val_direct.labels_

        train_counts = pd.Series(labels_train).value_counts().sort_index()
        train_proportions = train_counts / len(x_train)
        expected_val_counts = train_proportions * len(x_val)

        observed_val_counts = (
            pd.Series(labels_val_predicted)
            .value_counts()
            .reindex(range(params["n_clusters"]), fill_value=0)
            .sort_index()
        )

        chi2_stat, p_value = self._calcular_qui_quadrado(
            observed_val_counts, expected_val_counts
        )
        ari_score = adjusted_rand_score(
            labels_val_predicted, labels_val_direct
        )

        msn_resul = self._formatar_resultado(
            titulo="K-means",
            labels_train=labels_train,
            expected_val_counts=expected_val_counts,
            observed_val_counts=observed_val_counts,
            chi2_stat=chi2_stat,
            p_value=p_value,
            ari_score=ari_score,
        )
        grafico = self.grafico_scatter(
            hue=labels_val_predicted, df_grafico=df_test
        )

        return grafico, msn_resul

    def __testar_dbscan(
        self, best_param: dict, df_test: pd.DataFrame
    ) -> tuple[sns.JointGrid, str]:
        """
        Testa a consistência dos clusters DBSCAN entre treino e
        validação/teste via Qui-Quadrado de Aderência e ARI.
        """
        x_train, _, _ = self._preparar_par(self.df)
        x_val, _, _ = self._preparar_par(df_test)

        dbscan_train = DBSCAN(**best_param).fit(x_train)
        labels_train = dbscan_train.labels_

        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(x_train, labels_train)
        labels_val_predicted = knn.predict(x_val)

        dbscan_val_direct = DBSCAN(**best_param).fit(x_val)
        labels_val_direct = dbscan_val_direct.labels_

        all_clusters = np.unique(
            np.concatenate([labels_train, labels_val_predicted])
        )

        train_counts = (
            pd.Series(labels_train)
            .value_counts()
            .reindex(all_clusters, fill_value=0)
            .sort_index()
        )
        train_proportions = train_counts / len(x_train)
        expected_val_counts = train_proportions * len(x_val)

        observed_val_counts = (
            pd.Series(labels_val_predicted)
            .value_counts()
            .reindex(all_clusters, fill_value=0)
            .sort_index()
        )

        chi2_stat, p_value = self._calcular_qui_quadrado(
            observed_val_counts, expected_val_counts
        )
        ari_score = adjusted_rand_score(
            labels_val_predicted, labels_val_direct
        )

        msn_resul = self._formatar_resultado(
            titulo="DBSCAN",
            labels_train=labels_train,
            expected_val_counts=expected_val_counts,
            observed_val_counts=observed_val_counts,
            chi2_stat=chi2_stat,
            p_value=p_value,
            ari_score=ari_score,
        )
        grafico = self.grafico_scatter(
            hue=labels_val_predicted, df_grafico=df_test
        )

        return grafico, msn_resul

    def __testar_agg(
        self, best_param: dict, df_test: pd.DataFrame
    ) -> tuple[sns.JointGrid, str]:
        """
        Testa a consistência dos clusters Agglomerative entre treino e
        validação/teste via Qui-Quadrado de Aderência e ARI.
        """
        x_train, _, _ = self._preparar_par(self.df)
        x_val, _, _ = self._preparar_par(df_test)

        agg_train = AgglomerativeClustering(**best_param)
        labels_train = agg_train.fit_predict(x_train)

        if len(np.unique(labels_train)) > 1:
            clf_centroid = NearestCentroid()
            clf_centroid.fit(x_train, labels_train)
            labels_val_predicted = clf_centroid.predict(x_val)
        else:
            labels_val_predicted = np.full(len(x_val), labels_train[0])

        agg_val_direct = AgglomerativeClustering(**best_param)
        labels_val_direct = agg_val_direct.fit_predict(x_val)

        all_clusters = np.unique(labels_train)

        train_counts = (
            pd.Series(labels_train)
            .value_counts()
            .reindex(all_clusters, fill_value=0)
            .sort_index()
        )
        train_proportions = train_counts / len(x_train)
        expected_val_counts = train_proportions * len(x_val)

        observed_val_counts = (
            pd.Series(labels_val_predicted)
            .value_counts()
            .reindex(all_clusters, fill_value=0)
            .sort_index()
        )

        chi2_stat, p_value = self._calcular_qui_quadrado(
            observed_val_counts, expected_val_counts
        )
        ari_score = adjusted_rand_score(
            labels_val_predicted, labels_val_direct
        )

        msn_resul = self._formatar_resultado(
            titulo="AGGLOMERATIVE",
            labels_train=labels_train,
            expected_val_counts=expected_val_counts,
            observed_val_counts=observed_val_counts,
            chi2_stat=chi2_stat,
            p_value=p_value,
            ari_score=ari_score,
        )
        grafico = self.grafico_scatter(
            hue=labels_val_predicted, df_grafico=df_test
        )

        return grafico, msn_resul

    def testar_modelo(
        self, best_param: dict, df_test: pd.DataFrame
    ) -> tuple[sns.JointGrid, str]:
        """
        Função para testar a consistência dos clusters obtidos no conjunto
        de treino com os clusters obtidos no conjunto de validação/teste.
        """
        if self.modelo == "kmeans":
            grafico, msn_resul = self.__testar_kmeans(
                best_param=best_param, df_test=df_test
            )
        elif self.modelo == "dbscan":
            grafico, msn_resul = self.__testar_dbscan(
                best_param=best_param, df_test=df_test
            )
        elif self.modelo == "agglomerative":
            grafico, msn_resul = self.__testar_agg(
                best_param=best_param, df_test=df_test
            )
        else:
            raise ValueError(f"Modelo desconhecido: {self.modelo}")

        return grafico, msn_resul
