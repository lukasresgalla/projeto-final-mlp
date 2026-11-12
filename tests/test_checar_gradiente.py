import unittest

import numpy as np

from checar_gradiente import checar_gradiente
from mlp.ativacoes import Sigmoid
from mlp.custos import MSE


class DensaMinima:
    def __init__(self, dimensao_entrada, dimensao_saida, gerador_aleatorio, bug=False):
        self.pesos = gerador_aleatorio.normal(
            0.0, 0.15, size=(dimensao_entrada, dimensao_saida)
        ).astype(np.float64)
        self.vieses = np.zeros((1, dimensao_saida), dtype=np.float64)
        self.bug = bug

    def passo_a_frente(self, entradas):
        self.entradas = np.asarray(entradas, dtype=np.float64).copy()
        return self.entradas @ self.pesos + self.vieses

    def passo_atras(self, gradiente_saida):
        self.grad_pesos = self.entradas.T @ gradiente_saida
        if self.bug:
            self.grad_pesos *= 2.0
        self.grad_vieses = np.sum(gradiente_saida, axis=0, keepdims=True)
        return gradiente_saida @ self.pesos.T


class RedeMinima:
    def __init__(self, camadas, funcao_custo):
        self.camadas = camadas
        self.funcao_custo = funcao_custo

    def passo_a_frente(self, entradas):
        saida = np.asarray(entradas, dtype=np.float64)
        for camada in self.camadas:
            saida = camada.passo_a_frente(saida)
        return saida

    def passo_atras(self, gradiente_saida):
        gradiente = gradiente_saida
        for camada in reversed(self.camadas):
            gradiente = camada.passo_atras(gradiente)


class TestChecarGradiente(unittest.TestCase):
    def setUp(self):
        gerador_aleatorio = np.random.RandomState(42)
        self.X = gerador_aleatorio.normal(size=(5, 3)).astype(np.float64)
        self.y = gerador_aleatorio.uniform(size=(5, 1)).astype(np.float64)

    def criar_rede(self, bug=False):
        gerador_aleatorio = np.random.RandomState(42)
        return RedeMinima(
            [
                DensaMinima(3, 4, gerador_aleatorio, bug=bug),
                Sigmoid(),
                DensaMinima(4, 1, gerador_aleatorio),
            ],
            MSE(),
        )

    def test_gradientes_corretos_passam(self):
        resultado = checar_gradiente(self.criar_rede(), self.X, self.y)
        self.assertTrue(resultado["passou"], resultado)
        self.assertLess(resultado["erro_maximo"], 1e-6)

    def test_gradiente_com_bug_e_detectado(self):
        resultado = checar_gradiente(self.criar_rede(bug=True), self.X, self.y)
        self.assertFalse(resultado["passou"], resultado)
        self.assertGreaterEqual(resultado["erro_maximo"], 1e-6)


if __name__ == "__main__":
    unittest.main()
