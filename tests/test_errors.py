import socket
import urllib.error

import pytest

from eggfotos.errors import friendly_error


@pytest.mark.parametrize("exc, trecho", [
    (MemoryError(), "Memória insuficiente"),
    (PermissionError("/x"), "Sem permissão"),
    (FileNotFoundError("/x/foto.jpg"), "não encontrado"),
    (urllib.error.URLError("boom"), "conexão"),
    (socket.timeout("t"), "conexão"),
    (OSError("cannot identify image file 'a.jpg'"), "não é uma imagem válida"),
    (OSError("image file is truncated (3 bytes not processed)"), "incompleta"),
    (OSError("disco cheio"), "ler ou gravar"),
    (ValueError("Qualidade deve ficar entre 1 e 100."), "Qualidade deve ficar"),   # já em português
    (ValueError("operands could not be broadcast together"), "erro inesperado"),
    (KeyError("x"), "erro inesperado"),
])
def test_friendly_error_in_portuguese(exc, trecho):
    assert trecho.lower() in friendly_error(exc).lower()


def test_unexpected_error_keeps_technical_detail():
    msg = friendly_error(RuntimeError("falha xyz"))
    assert "RuntimeError" in msg and "xyz" in msg
    assert "KeyError" in friendly_error(KeyError("k"))


def test_batch_log_uses_portuguese(tmp_path):
    from eggfotos.pipeline.recipe import Recipe
    from eggfotos.pipeline.runner import process_one
    r = Recipe("x", [{"op": "girar", "angulo": 90}])
    res = process_one(str(tmp_path / "nao_existe.jpg"), str(tmp_path / "o.jpg"), r.to_dict())
    assert res.status == "erro" and "não encontrado" in res.mensagem.lower()
