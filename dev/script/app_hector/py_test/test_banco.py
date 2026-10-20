from PIL import Image
from src.acesso_data import Data


def test_busca_dados_cluster():
    obj = Data()
    dados = obj.busca_dados_cluster(obj.dir_data)
    assert isinstance(dados, dict)
    assert "01_original" in dados
    assert "01_preparado" in dados
    assert "02_cluster" in dados


def test_busca_dados_class():
    obj = Data()
    dados = obj.busca_dados_class(obj.dir_data)
    assert isinstance(dados, dict)
    assert "03_dados_prep" in dados
    assert "04_metricas_treino" in dados
    assert "04_test_resultados" in dados


def test_busca_dados_regr():
    obj = Data()
    dados = obj.busca_dados_regr(obj.dir_data)
    assert isinstance(dados, dict)
    assert "04_dados_prep" in dados
    assert "04_metricas_treino" in dados
    assert "04_train_resultados" in dados
    assert "04_test_resultados" in dados


def test_imagens():
    obj = Data()
    logo = obj.logo_datalab()
    assert logo is None or isinstance(logo, Image.Image)
    assert isinstance(obj.icone_pagina(), Image.Image)
