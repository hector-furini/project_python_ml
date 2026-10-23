import os


def acesso_data_test() -> str:
    """
    Função para acessar o logo da aplicação.
    """
    path_img_local = "./app_hector/data/img/DataLab_Logo_i.jpg"
    path_img = "./dev/script/app_hector/data/img/DataLab_Logo_i.jpg"
    path_img_cwd = "./data/img/DataLab_Logo_i.jpg"

    if os.path.isfile(path_img_local):
        path_img = path_img_local
    elif not os.path.isfile(path_img) and os.path.isfile(path_img_cwd):
        path_img = path_img_cwd

    return path_img


def acesso_data_icon() -> str:
    """
    Função para acessar o ícone da aplicação.
    """
    path_img_local = "./app_hector/data/img/PythonML.ico"
    path_img = "./dev/script/app_hector/data/img/PythonML.ico"
    path_img_cwd = "./data/img/PythonML.ico"

    if os.path.isfile(path_img_local):
        path_img = path_img_local
    elif not os.path.isfile(path_img) and os.path.isfile(path_img_cwd):
        path_img = path_img_cwd

    return path_img


def acesso_data_banco() -> str:
    """
    Função para acessar o banco de dados de treino.
    """
    path_banco_local = "./app_hector/data/"
    path_banco = "./dev/script/app_hector/data/"
    path_banco_cwd = "./data/"

    if os.path.isdir(path_banco_local):
        path_banco = path_banco_local
    elif not os.path.isdir(path_banco) and os.path.isdir(path_banco_cwd):
        path_banco = path_banco_cwd

    return path_banco
