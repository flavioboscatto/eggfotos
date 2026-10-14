"""Mensagens de erro em português.

Bibliotecas (Pillow, OpenCV, rede, onnxruntime…) levantam erros em inglês técnico. Esta função
traduz os casos comuns e mantém o texto original entre parênteses para quem precisar depurar.
Sem Qt: serve também para o log do lote (processos).
"""
from __future__ import annotations

import socket
import ssl
import urllib.error


def _tecnico(e: BaseException) -> str:
    texto = " ".join(str(e).split())
    texto = texto if len(texto) <= 240 else texto[:237] + "…"
    return f"{type(e).__name__}: {texto}" if texto else type(e).__name__


def friendly_error(e: BaseException) -> str:
    """Resumo em português do erro `e` (+ detalhe técnico quando útil)."""
    nome = type(e).__name__
    modulo = type(e).__module__ or ""
    texto = str(e)
    baixo = texto.lower()

    if isinstance(e, MemoryError):
        return "Memória insuficiente para esta foto. Feche outros programas ou use uma foto menor."
    if isinstance(e, PermissionError):
        return f"Sem permissão para acessar o arquivo ou a pasta. ({texto})"
    if isinstance(e, FileNotFoundError):
        return f"Arquivo ou pasta não encontrado. ({texto})"
    if isinstance(e, IsADirectoryError):
        return f"Era esperado um arquivo, mas é uma pasta. ({texto})"
    if isinstance(e, (urllib.error.URLError, socket.timeout, TimeoutError, ssl.SSLError, ConnectionError)):
        return ("Falha de conexão ao baixar o arquivo. Verifique a internet e tente de novo. "
                f"({_tecnico(e)})")
    if nome == "UnidentifiedImageError" or "cannot identify image file" in baixo:
        return "O arquivo não é uma imagem válida ou está corrompido."
    if "image file is truncated" in baixo or "broken data stream" in baixo:
        return "A imagem está incompleta ou corrompida."
    if "decompression bomb" in baixo:
        return "A imagem é grande demais para ser aberta com segurança."
    if modulo.startswith("cv2") or nome == "error" and "cv2" in modulo:
        return f"Erro no processamento da imagem. ({_tecnico(e)})"
    if modulo.startswith("onnxruntime"):
        return f"Erro ao executar o modelo de IA. ({_tecnico(e)})"
    if isinstance(e, OSError):
        return f"Erro ao ler ou gravar o arquivo. ({_tecnico(e)})"
    if isinstance(e, (RuntimeError, ValueError)) and _parece_portugues(texto):
        return texto  # as mensagens do próprio programa já estão em português
    return f"Ocorreu um erro inesperado. ({_tecnico(e)})"


_PT_PISTAS = (" não ", " de ", " da ", " do ", " para ", " foi ", "ção", "ões", " precisa", " muito",
              " tente", " baixe", " marc", " modelo", " foto", " mínimo", " máximo", " inválid")


def _parece_portugues(texto: str) -> bool:
    t = f" {texto.lower()} "
    return any(p in t for p in _PT_PISTAS) or any(c in t for c in "áàâãéêíóôõúç")
