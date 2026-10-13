"""Núcleo de processamento: funções puras sobre arrays NumPy, sem dependência de Qt.

Convenção de imagem em todo o projeto:
    np.ndarray uint8, formato (altura, largura, 3) RGB ou (altura, largura, 4) RGBA.
Toda operação recebe uma imagem e devolve uma NOVA imagem (nunca altera a entrada).
"""
