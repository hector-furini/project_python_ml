import streamlit as st
from pydantic import validate_call
from src.iniciar import Iniciar


class Paginas:
    @validate_call
    def estruturar_pagina(self) -> str:
        """
        Função de estruturação da página inicial.
        Parametros:
            None
        Retorna:
            str: Mensagem de sucesso.
        """
        mensagem = "Alguma coisa"
        inicia = Iniciar(mensagem=mensagem)

        st.set_page_config(
            page_title=inicia.testo,
            layout="wide",
        )
        st.markdown(
            """
            <style>
                .block-container {
                        padding-top: 0rem;
                        padding-bottom: 0rem;
                        padding-left: 3rem;
                        padding-right: 3rem;
                    }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # title = "Métricas de Modelos de Machine Learning"
        st.markdown(
            f"<br><h3 style='text-align: center; '>{inicia.testo}</h3>",
            unsafe_allow_html=True,
        )

        return "Ajustado"
