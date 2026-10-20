import seaborn as sns
from src.acesso_data import Data
from src.modelos.clustering import Clustering
from src.paginas import Paginas


def test_criar_paginas():
    obj = Paginas()
    assert isinstance(obj.data, Data)
    assert len(obj.paginas) == 5
    assert len(obj.modelos_cluster) == 5
    assert len(obj.modelos_class) == 14
    assert len(obj.modelos_regr) == 12


def test_clustering():
    obj = Paginas()
    cluster = Clustering(obj.dados_cluster)
    fig = cluster.testar_modelo("K-means")
    assert isinstance(fig, sns.axisgrid.JointGrid)
