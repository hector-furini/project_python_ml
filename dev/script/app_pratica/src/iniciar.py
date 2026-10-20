from pydantic import validate_call


class Iniciar:
    @validate_call
    def __init__(self, mensagem: str) -> None:
        self.testo = mensagem
        return None

    def __mensagem_boas_vindas(self) -> str:
        """
        Função de inicialização do sistema
        Parametros:
            mensagem (str): Mensagem recebida
        Retorno:
            retorno (str): Mensagem transmitida
        """
        retorno = self.testo
        return retorno

    @validate_call
    def mostra_mensagem(self) -> str:
        print("Mensagem de boas vindas")
        print(f"{self.__mensagem_boas_vindas()}")
        return "Mensagem enviada"
