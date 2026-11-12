import numpy as np
from mlp.base import Inicializador


class Xavier(Inicializador):
    def inicializar(
        self,
        formato: tuple[int, int],
        gerador_aleatorio: np.random.RandomState,
    ) -> np.ndarray:
        dimensao_entrada, dimensao_saida = formato
        limite = np.sqrt(6.0 / (dimensao_entrada + dimensao_saida))
        return gerador_aleatorio.uniform(-limite, limite, size=formato)


class He(Inicializador):
    def inicializar(
        self,
        formato: tuple[int, int],
        gerador_aleatorio: np.random.RandomState,
    ) -> np.ndarray:
        dimensao_entrada = formato[0]
        desvio = np.sqrt(2.0 / dimensao_entrada)
        return gerador_aleatorio.normal(0.0, desvio, size=formato)


class NormalPequena(Inicializador):
    def inicializar(
        self,
        formato: tuple[int, int],
        gerador_aleatorio: np.random.RandomState,
    ) -> np.ndarray:
        return gerador_aleatorio.normal(0.0, 0.01, size=formato)