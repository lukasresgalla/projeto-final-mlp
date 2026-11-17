import unittest

import numpy as np

from checar_gradiente import checar_gradiente
from mlp.ativacoes import Sigmoid
from mlp.camadas import Densa
from mlp.custos import MSE
from mlp.rede import RedeNeural


class TestRedeNeural(unittest.TestCase):
    def criar_rede_xor(self):
        rede = RedeNeural(semente_aleatoria=42)
        rede.adicionar(Densa(2, 4, gerador_aleatorio=np.random.RandomState(1)))
        rede.adicionar(Sigmoid())
        rede.adicionar(Densa(4, 1, gerador_aleatorio=np.random.RandomState(2)))
        rede.adicionar(Sigmoid())
        rede.compilar(MSE())
        return rede

    def test_rede_xor_reduz_custo_com_atualizacoes_manuais(self):
        X = np.array(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=float
        )
        y = np.array([[0.0], [1.0], [1.0], [0.0]], dtype=float)
        rede = self.criar_rede_xor()
        custo_inicial = rede.funcao_custo.calcular_custo(y, rede.predizer(X))

        for _ in range(8000):
            saida = rede.passo_a_frente(X)
            gradiente = rede.funcao_custo.gradiente(y, saida)
            rede.passo_atras(gradiente)
            for camada in rede.camadas:
                if hasattr(camada, "atualizar"):
                    camada.atualizar(1.0)

        custo_final = rede.funcao_custo.calcular_custo(y, rede.predizer(X))
        self.assertLess(custo_final, custo_inicial)
        self.assertLess(custo_final, 0.02)

    def test_gradientes_da_rede_passam_na_verificacao_numerica(self):
        X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=float)
        y = np.array([[0.0], [1.0], [1.0]], dtype=float)

        resultado = checar_gradiente(self.criar_rede_xor(), X, y)

        self.assertTrue(resultado["passou"], resultado)
        self.assertLess(resultado["erro_maximo"], 1e-6)


if __name__ == "__main__":
    unittest.main()
