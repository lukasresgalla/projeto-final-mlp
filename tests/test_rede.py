import contextlib
import io
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

    def dados_xor(self):
        X = np.array(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=float
        )
        y = np.array([[0.0], [1.0], [1.0], [0.0]], dtype=float)
        return X, y

    def test_treinar_devolve_historico_com_uma_entrada_por_epoca(self):
        X, y = self.dados_xor()
        rede = self.criar_rede_xor()

        historico = rede.treinar(X, y, epocas=5, taxa_aprendizagem=1.0, verbal=False)

        self.assertEqual(set(historico), {"custo_treino", "custo_validacao"})
        self.assertEqual(len(historico["custo_treino"]), 5)
        self.assertEqual(historico["custo_validacao"], [])

    def test_treinar_com_validacao_registra_custo_de_validacao(self):
        X, y = self.dados_xor()
        rede = self.criar_rede_xor()

        historico = rede.treinar(
            X, y, epocas=4, taxa_aprendizagem=1.0,
            X_validacao=X, y_validacao=y, verbal=False,
        )

        self.assertEqual(len(historico["custo_treino"]), 4)
        self.assertEqual(len(historico["custo_validacao"]), 4)

    def test_treinar_reduz_custo_no_xor_com_lote_menor_que_o_conjunto(self):
        X, y = self.dados_xor()
        rede = self.criar_rede_xor()

        # lote 3 com m=4: lotes de tamanho 3 e 1 (último lote menor)
        historico = rede.treinar(
            X, y, epocas=3000, taxa_aprendizagem=1.0, tamanho_lote=3, verbal=False
        )

        custos = historico["custo_treino"]
        self.assertLess(custos[-1], custos[0])
        self.assertLess(custos[-1], 0.02)

    def test_treinar_e_reprodutivel_com_a_mesma_semente(self):
        X, y = self.dados_xor()

        h1 = self.criar_rede_xor().treinar(
            X, y, epocas=20, taxa_aprendizagem=1.0, tamanho_lote=2, verbal=False
        )
        h2 = self.criar_rede_xor().treinar(
            X, y, epocas=20, taxa_aprendizagem=1.0, tamanho_lote=2, verbal=False
        )

        self.assertEqual(h1["custo_treino"], h2["custo_treino"])

    def test_treinar_imprime_a_cada_10_epocas_quando_verbal(self):
        X, y = self.dados_xor()
        rede = self.criar_rede_xor()

        with contextlib.redirect_stdout(io.StringIO()) as saida:
            rede.treinar(X, y, epocas=25, taxa_aprendizagem=1.0, verbal=True)

        linhas = saida.getvalue().strip().splitlines()
        self.assertEqual(len(linhas), 2)  # épocas 10 e 20


if __name__ == "__main__":
    unittest.main()
