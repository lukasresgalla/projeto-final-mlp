import unittest

import numpy as np

from mlp.camadas import Densa


class TestDensa(unittest.TestCase):
    def setUp(self):
        self.camada = Densa(2, 2, gerador_aleatorio=np.random.RandomState(42))
        self.camada.pesos = np.array([[0.2, -0.4], [0.7, 0.3]])
        self.camada.vieses = np.array([[0.1, -0.2]])
        self.entradas = np.array([[1.0, 2.0], [-1.0, 0.5]])

    def test_inicializa_com_formatos_e_vieses_zerados(self):
        camada = Densa(3, 4, gerador_aleatorio=np.random.RandomState(1))
        self.assertEqual(camada.pesos.shape, (3, 4))
        self.assertEqual(camada.vieses.shape, (1, 4))
        np.testing.assert_array_equal(camada.vieses, np.zeros((1, 4)))

    def test_passo_a_frente(self):
        saida = self.camada.passo_a_frente(self.entradas)
        esperado = self.entradas @ self.camada.pesos + self.camada.vieses
        np.testing.assert_allclose(saida, esperado)

    def test_passo_atras_calcula_gradientes_e_retorna_gradiente_anterior(self):
        self.camada.passo_a_frente(self.entradas)
        gradiente_saida = np.array([[0.5, -0.3], [0.2, 0.8]])

        gradiente_entrada = self.camada.passo_atras(gradiente_saida)

        np.testing.assert_allclose(
            self.camada.grad_pesos, self.entradas.T @ gradiente_saida
        )
        np.testing.assert_allclose(
            self.camada.grad_vieses,
            np.sum(gradiente_saida, axis=0, keepdims=True),
        )
        np.testing.assert_allclose(
            gradiente_entrada, gradiente_saida @ self.camada.pesos.T
        )

    def test_atualizar_altera_pesos_e_vieses_na_direcao_do_gradiente(self):
        self.camada.passo_a_frente(self.entradas)
        self.camada.passo_atras(np.array([[0.5, -0.3], [0.2, 0.8]]))
        pesos_antes = self.camada.pesos.copy()
        vieses_antes = self.camada.vieses.copy()
        taxa = 0.1

        self.camada.atualizar(taxa)

        np.testing.assert_allclose(
            self.camada.pesos, pesos_antes - taxa * self.camada.grad_pesos
        )
        np.testing.assert_allclose(
            self.camada.vieses, vieses_antes - taxa * self.camada.grad_vieses
        )


if __name__ == "__main__":
    unittest.main()
