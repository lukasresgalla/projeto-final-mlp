import numpy as np
from mlp.base import Ativacao, Camada, FuncaoCusto


class RedeNeural:
    def __init__(self, semente_aleatoria: int | None = None):
        self.camadas: list[Camada | Ativacao] = []
        self.funcao_custo: FuncaoCusto | None = None
        self.semente_aleatoria = semente_aleatoria
        self.gerador_aleatorio = np.random.RandomState(semente_aleatoria)

    def adicionar(self, camada: Camada | Ativacao) -> None:
        self.camadas.append(camada)

    def compilar(self, funcao_custo: FuncaoCusto) -> None:
        self.funcao_custo = funcao_custo

    def passo_a_frente(self, X: np.ndarray) -> np.ndarray:
        saida = X
        for camada in self.camadas:
            saida = camada.passo_a_frente(saida)
        return saida

    def passo_atras(self, gradiente_saida: np.ndarray) -> None:
        gradiente = gradiente_saida
        for camada in reversed(self.camadas):
            gradiente = camada.passo_atras(gradiente)

    def treinar(
        self,
        X_treino: np.ndarray,
        y_treino: np.ndarray,
        epocas: int,
        taxa_aprendizagem: float,
        tamanho_lote: int = 32,
        X_validacao: np.ndarray | None = None,
        y_validacao: np.ndarray | None = None,
        verbal: bool = True,
    ) -> dict[str, list[float]]:
        m = X_treino.shape[0]

        for epoca in range(epocas):
            indices = self.gerador_aleatorio.permutation(m)

            for inicio in range(0, m, tamanho_lote):
                indices_lote = indices[inicio:inicio + tamanho_lote]
                Xb = X_treino[indices_lote]
                yb = y_treino[indices_lote]

        raise NotImplementedError

    def predizer(self, X: np.ndarray) -> np.ndarray:
        return self.passo_a_frente(X)
