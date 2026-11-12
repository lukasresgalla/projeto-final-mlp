import unittest

import numpy as np

from mlp.custos import (
    EntropiaCruzadaBinaria,
    EntropiaCruzadaCategorica,
    MSE,
)


class TestCustos(unittest.TestCase):
    def verificar_derivada_numerica(self, funcao_custo, y_verdadeiro, y_predito):
        passo = 1e-5
        gradiente_numerico = np.zeros_like(y_predito)

        for indice in np.ndindex(y_predito.shape):
            p_mais = y_predito.copy()
            p_menos = y_predito.copy()
            p_mais[indice] += passo
            p_menos[indice] -= passo
            custo_mais = funcao_custo.calcular_custo(y_verdadeiro, p_mais)
            custo_menos = funcao_custo.calcular_custo(y_verdadeiro, p_menos)
            gradiente_numerico[indice] = (custo_mais - custo_menos) / (2.0 * passo)

        gradiente = funcao_custo.gradiente(y_verdadeiro, y_predito)
        self.assertEqual(gradiente.shape, y_predito.shape)
        np.testing.assert_allclose(
            gradiente,
            gradiente_numerico,
            rtol=1e-5,
            atol=1e-7,
        )

    def test_derivada_numerica_mse(self):
        y = np.array([[0.4, -0.2], [1.1, 0.7]])
        p = np.array([[0.2, 0.3], [0.8, 0.5]])
        self.verificar_derivada_numerica(MSE(), y, p)

    def test_derivada_numerica_entropia_binaria(self):
        y = np.array([[1.0], [0.0], [1.0]])
        p = np.array([[0.72], [0.31], [0.84]])
        self.verificar_derivada_numerica(EntropiaCruzadaBinaria(), y, p)

    def test_derivada_numerica_entropia_categorica(self):
        y = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        p = np.array([[0.62, 0.28, 0.10], [0.17, 0.56, 0.27]])
        np.testing.assert_allclose(np.sum(p, axis=1), np.ones(p.shape[0]))
        self.verificar_derivada_numerica(EntropiaCruzadaCategorica(), y, p)

    def test_mse_zero_quando_predicao_igual_ao_alvo(self):
        y = np.array([[0.5], [-1.2], [2.4]])
        self.assertEqual(MSE().calcular_custo(y, y.copy()), 0.0)

    def test_entropias_finitas_nos_extremos(self):
        y_binaria = np.array([[0.0], [1.0]])
        p_binaria = np.array([[0.0], [1.0]])
        custo_binario = EntropiaCruzadaBinaria()
        self.assertTrue(np.isfinite(custo_binario.calcular_custo(y_binaria, p_binaria)))
        self.assertTrue(np.all(np.isfinite(custo_binario.gradiente(y_binaria, p_binaria))))

        y_categorica = np.array([[1.0, 0.0], [0.0, 1.0]])
        p_categorica = np.array([[1.0, 0.0], [0.0, 1.0]])
        custo_categorico = EntropiaCruzadaCategorica()
        self.assertTrue(
            np.isfinite(custo_categorico.calcular_custo(y_categorica, p_categorica))
        )
        self.assertTrue(
            np.all(np.isfinite(custo_categorico.gradiente(y_categorica, p_categorica)))
        )

    def test_custos_devolvem_float_python(self):
        y_regressao = np.array([[0.0], [1.0]])
        p_regressao = np.array([[0.2], [0.8]])
        y_binaria = np.array([[0.0], [1.0]])
        p_binaria = np.array([[0.2], [0.8]])
        y_categorica = np.array([[1.0, 0.0], [0.0, 1.0]])
        p_categorica = np.array([[0.7, 0.3], [0.2, 0.8]])
        casos = [
            (MSE(), y_regressao, p_regressao),
            (EntropiaCruzadaBinaria(), y_binaria, p_binaria),
            (EntropiaCruzadaCategorica(), y_categorica, p_categorica),
        ]

        for custo, y, p in casos:
            with self.subTest(custo=type(custo).__name__):
                self.assertIs(type(custo.calcular_custo(y, p)), float)


if __name__ == "__main__":
    unittest.main()