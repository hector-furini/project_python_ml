from src.iniciar import Iniciar


def test_atributo_testo():
    mensagem = "Alguma coisa"

    inicia = Iniciar(mensagem=mensagem)

    assert inicia.testo == mensagem


def test_mostra_mensagem():
    testo_inicial = "Mensagem enviada"

    inicia = Iniciar(mensagem=testo_inicial)
    testo_saida = inicia.mostra_mensagem()

    assert testo_inicial == testo_saida
    assert testo_saida != 10
