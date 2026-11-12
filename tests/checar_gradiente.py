import numpy as np


def checar_gradiente(rede, X, y, h=1e-5, tolerancia=1e-6):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    saida = rede.passo_a_frente(X)
    rede.funcao_custo.calcular_custo(y, saida)
    gradiente_saida = rede.funcao_custo.gradiente(y, saida)
    rede.passo_atras(gradiente_saida)

    gradientes_analiticos = {}
    for indice, camada in enumerate(rede.camadas):
        gradientes_camada = {}
        if hasattr(camada, "grad_pesos"):
            gradientes_camada["grad_pesos"] = np.copy(camada.grad_pesos)
        if hasattr(camada, "grad_vieses"):
            gradientes_camada["grad_vieses"] = np.copy(camada.grad_vieses)
        if gradientes_camada:
            gradientes_analiticos[indice] = gradientes_camada

    erro_maximo = 0.0
    for indice, camada in enumerate(rede.camadas):
        for nome_parametro, nome_gradiente in (
            ("pesos", "grad_pesos"),
            ("vieses", "grad_vieses"),
        ):
            if not hasattr(camada, nome_parametro):
                continue

            parametro = getattr(camada, nome_parametro)
            gradiente_analitico = gradientes_analiticos[indice][nome_gradiente]
            for posicao in np.ndindex(parametro.shape):
                valor_original = parametro[posicao].copy()
                try:
                    parametro[posicao] = valor_original + h
                    custo_mais = rede.funcao_custo.calcular_custo(
                        y, rede.passo_a_frente(X)
                    )
                    parametro[posicao] = valor_original - h
                    custo_menos = rede.funcao_custo.calcular_custo(
                        y, rede.passo_a_frente(X)
                    )
                finally:
                    parametro[posicao] = valor_original

                gradiente_numerico = (custo_mais - custo_menos) / (2.0 * h)
                gradiente = gradiente_analitico[posicao]
                erro_relativo = abs(gradiente - gradiente_numerico) / max(
                    abs(gradiente) + abs(gradiente_numerico), 1e-12
                )
                erro_maximo = max(erro_maximo, float(erro_relativo))

    return {"erro_maximo": erro_maximo, "passou": erro_maximo < tolerancia}
