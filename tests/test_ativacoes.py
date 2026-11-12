import unittest

import numpy as np

from mlp.ativacoes import (
    ELU,
    HardSigmoid,
    LeakyReLU,
    Linear,
    ReLU,
    Sigmoid,
    Softmax,
    Softplus,
    Softsign,
    Swish,
)


class TestDerivadasAtivacoes(unittest.TestCase):
    def verificar_derivada_numerica(self, ativacao, entradas, gradiente_saida):
        passo = 1e-5
        gradiente_numerico = np.zeros_like(entradas)

        for indice in np.ndindex(entradas.shape):
            entradas_mais = entradas.copy()
            entradas_menos = entradas.copy()
            entradas_mais[indice] += passo
            entradas_menos[indice] -= passo

            custo_mais = np.sum(
                ativacao.passo_a_frente(entradas_mais) * gradiente_saida
            )
            custo_menos = np.sum(
                ativacao.passo_a_frente(entradas_menos) * gradiente_saida
            )
            gradiente_numerico[indice] = (custo_mais - custo_menos) / (2.0 * passo)

        ativacao.passo_a_frente(entradas)
        gradiente_calculado = ativacao.passo_atras(gradiente_saida)
        np.testing.assert_allclose(
            gradiente_calculado,
            gradiente_numerico,
            rtol=1e-5,
            atol=1e-7,
        )

    def test_derivadas_elemento_a_elemento(self):
        entradas = np.array([[-2.2, -0.8, 0.7], [1.4, 2.6, -1.3]])
        gradiente_saida = np.array([[0.3, -0.6, 0.9], [-0.2, 0.5, 0.8]])
        ativacoes = [
            Sigmoid(),
            ReLU(),
            LeakyReLU(),
            ELU(),
            Softplus(),
            Softsign(),
            Swish(),
            HardSigmoid(),
            Linear(),
        ]

        for ativacao in ativacoes:
            with self.subTest(ativacao=type(ativacao).__name__):
                self.verificar_derivada_numerica(
                    ativacao, entradas, gradiente_saida
                )

    def test_derivada_softmax(self):
        entradas = np.array([[-1.2, 0.4, 2.1], [1.7, -0.6, 0.2]])
        gradiente_saida = np.array([[0.3, -0.7, 0.2], [-0.4, 0.8, 0.6]])
        self.verificar_derivada_numerica(Softmax(), entradas, gradiente_saida)

    def test_softmax_estavel_e_normalizada_por_linha(self):
        saida = Softmax().passo_a_frente(np.array([[1000.0, 1001.0, 999.0]]))
        self.assertTrue(np.all(np.isfinite(saida)))
        np.testing.assert_allclose(np.sum(saida, axis=1), np.array([1.0]))


if __name__ == "__main__":
    unittest.main()