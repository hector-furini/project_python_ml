import streamlit as st
from src.paginas import Paginas


@st.cache_resource
def setup():
    return Paginas()


if __name__ == "__main__":
    app = setup()
    app.mostrar_pagina()
