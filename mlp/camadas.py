import numpy as np
from mlp.base import Camada, Inicializador
from mlp.inicializadores import Xavier


class Densa(Camada):
    def __init__(
        self,
        dimensao_entrada: int,
        dimensao_saida: int,
        inicializador: Inicializador | None = None,
        gerador_aleatorio: np.random.RandomState | None = None,
    ):
        super().__init__()
        self.dimensao_entrada = dimensao_entrada
        self.dimensao_saida = dimensao_saida
        self.gerador_aleatorio = (
            gerador_aleatorio
            if gerador_aleatorio is not None
            else np.random.RandomState()
        )
        inicializador = inicializador if inicializador is not None else Xavier()
        self.pesos = inicializador.inicializar(
            (dimensao_entrada, dimensao_saida), self.gerador_aleatorio
        )
        self.vieses = np.zeros((1, dimensao_saida), dtype=float)
        self.grad_pesos = np.zeros_like(self.pesos)
        self.grad_vieses = np.zeros_like(self.vieses)

    def passo_a_frente(self, entradas: np.ndarray) -> np.ndarray:
        self.entradas = np.array(entradas, copy=True)
        return self.entradas @ self.pesos + self.vieses

    def passo_atras(self, gradiente_saida: np.ndarray) -> np.ndarray:
        self.grad_pesos = self.entradas.T @ gradiente_saida
        self.grad_vieses = np.sum(gradiente_saida, axis=0, keepdims=True)
        return gradiente_saida @ self.pesos.T

    def atualizar(self, taxa_aprendizagem: float) -> None:
        self.pesos -= taxa_aprendizagem * self.grad_pesos
        self.vieses -= taxa_aprendizagem * self.grad_vieses
