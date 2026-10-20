from main import setup
from src.paginas import Paginas


def test_setup():
    assert isinstance(setup(), Paginas)
