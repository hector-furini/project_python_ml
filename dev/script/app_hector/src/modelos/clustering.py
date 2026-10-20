import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class Clustering:
    """Classe para apresentar gráficos de clusterização"""

    def __init__(self, dados: pd.DataFrame):
        self.__dados = dados

    def __definir_cores(self):
        return [
            "#e41a1c",
            "#377eb8",
            "#4daf4a",
            "#984ea3",
            "#ff7f00",
            "#ffff33",
            "#a65628",
            "#f781bf",
            "#999999",
            "#e41a1c",
            "#377eb8",
            "#4daf4a",
            "#984ea3",
            "#ff7f00",
            "#ffff33",
            "#a65628",
            "#f781bf",
            "#999999",
        ]

    def grafico_scatter(
        self,
        dados_cluster: pd.DataFrame,
        clusters: list,
        col_x: str,
        col_y: str,
    ):
        cores = self.__definir_cores()
        color_dict = dict(zip(clusters, cores[0 : len(clusters)]))
        p = sns.jointplot(
            data=dados_cluster,
            x=col_x,
            y=col_y,
            hue="cluster",
            kind="scatter",
            palette=color_dict,
            height=4,
            s=10,
        )
        sns.move_legend(p.ax_joint, "upper left", bbox_to_anchor=(1.2, 1.2))
        return p

    def testar_modelo(self, nome_modelo: str):
        dados_cluster = self.__dados.copy()
        dados_cluster["cluster"] = self.__dados[nome_modelo]
        clusters = list(dados_cluster["cluster"].unique())
        clusters.sort()
        fig = self.grafico_scatter(
            dados_cluster, clusters, "comp1", "comp2"
        )
        plt.close(fig.fig)
        return fig
