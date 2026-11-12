import unittest

import numpy as np

from mlp.inicializadores import He, NormalPequena, Xavier


class TestInicializadores(unittest.TestCase):
    def setUp(self):
        self.inicializadores = [Xavier(), He(), NormalPequena()]

    def test_formato_solicitado(self):
        formato = (4, 3)
        for inicializador in self.inicializadores:
            with self.subTest(inicializador=type(inicializador).__name__):
                pesos = inicializador.inicializar(
                    formato, np.random.RandomState(42)
                )
                self.assertEqual(pesos.shape, formato)

    def test_mesma_semente_reproduz_e_sementes_diferentes_variam(self):
        formato = (4, 3)
        for inicializador in self.inicializadores:
            with self.subTest(inicializador=type(inicializador).__name__):
                pesos_a = inicializador.inicializar(
                    formato, np.random.RandomState(42)
                )
                pesos_b = inicializador.inicializar(
                    formato, np.random.RandomState(42)
                )
                pesos_c = inicializador.inicializar(
                    formato, np.random.RandomState(43)
                )
                np.testing.assert_array_equal(pesos_a, pesos_b)
                self.assertFalse(np.array_equal(pesos_a, pesos_c))

    def test_xavier_respeita_limites(self):
        formato = (120, 80)
        pesos = Xavier().inicializar(formato, np.random.RandomState(42))
        limite = np.sqrt(6.0 / sum(formato))
        self.assertTrue(np.all(pesos >= -limite))
        self.assertTrue(np.all(pesos <= limite))

    def test_he_tem_media_e_desvio_esperados(self):
        formato = (1000, 1000)
        esperado = np.sqrt(2.0 / formato[0])
        pesos = He().inicializar(formato, np.random.RandomState(42))
        self.assertLess(abs(float(np.mean(pesos))), 0.02 * esperado)
        self.assertAlmostEqual(float(np.std(pesos)), esperado, delta=0.02 * esperado)

    def test_normal_pequena_tem_media_e_desvio_esperados(self):
        formato = (1000, 1000)
        esperado = 0.01
        pesos = NormalPequena().inicializar(
            formato, np.random.RandomState(42)
        )
        self.assertLess(abs(float(np.mean(pesos))), 0.02 * esperado)
        self.assertAlmostEqual(float(np.std(pesos)), esperado, delta=0.02 * esperado)

    def test_valores_sao_finitos(self):
        formato = (100, 50)
        for inicializador in self.inicializadores:
            with self.subTest(inicializador=type(inicializador).__name__):
                pesos = inicializador.inicializar(
                    formato, np.random.RandomState(42)
                )
                self.assertTrue(np.all(np.isfinite(pesos)))


if __name__ == "__main__":
    unittest.main()