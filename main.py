"""EggFotos — ponto de entrada.

    python main.py            abre o editor
    python main.py foto.jpg   abre o editor já com a foto
"""
import multiprocessing
import sys


def main() -> int:
    from eggfotos.ui.app import run
    return run(sys.argv)


if __name__ == "__main__":
    # Obrigatório no Windows: o lote usa processos paralelos e, sem isto,
    # cada processo filho abriria uma nova janela do programa.
    multiprocessing.freeze_support()
    sys.exit(main())
