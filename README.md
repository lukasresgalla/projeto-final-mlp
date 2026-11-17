# Projeto Final: Biblioteca de Redes Neurais (MLP)

**INE5664 — Aprendizado de Máquina (UFSC)**

Implementação de uma biblioteca de Multilayer Perceptron (MLP) do zero, usando apenas NumPy, com experimentos de regressão, classificação binária e classificação multiclasse documentados no notebook `relatorio_tecnico.ipynb`.

**Equipe:**
- Lukas Dávila Resgalla — 23200519
- Leila Nunes dos Anjos — 23204361
- Giovanna Gomes Tonelli — 23205461

## Estrutura do projeto

```
mlp/
  base.py            # Classes abstratas (Camada, Ativacao, FuncaoCusto, Inicializador)
  camadas.py         # Camada Densa (forward/backward/atualização de pesos)
  ativacoes.py       # Sigmoid, ReLU, LeakyReLU, ELU, Softplus, Softsign, Swish, HardSigmoid, Softmax, Linear
  custos.py          # MSE, EntropiaCruzadaBinaria, EntropiaCruzadaCategorica
  inicializadores.py # Xavier, He, NormalPequena
  rede.py            # RedeNeural (monta as camadas e treina)
tests/
  checar_gradiente.py
  test_*.py          # Testes unitários de cada módulo
relatorio_tecnico.ipynb  # Notebook com os 3 experimentos (regressão, classificação binária e multiclasse)
```

## Regras do código-fonte (`mlp/`)

* **Contrato fixo (`mlp/base.py`):** as interfaces abstratas deste arquivo não são alteradas.
* **Restrição de bibliotecas:** o pacote `mlp/` usa exclusivamente NumPy. Nenhum framework de ML ou diferenciação automática é utilizado dentro da pasta `mlp/`.
* `pandas`, `scikit-learn` e `matplotlib` são usados somente no notebook, para carregar dados, dividir conjuntos, calcular métricas auxiliares e gerar gráficos — nunca para treinar o modelo.

## Como rodar

1. Criar e ativar o ambiente virtual:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Instalar as dependências:
   ```bash
   pip install numpy scikit-learn pandas matplotlib jupyter
   ```
3. Rodar os testes:
   ```bash
   PYTHONPATH=.:tests python -m unittest discover -s tests
   ```
4. Abrir o notebook `relatorio_tecnico.ipynb` no VS Code (extensão Jupyter), selecionar o kernel do `.venv` e executar todas as células, de cima para baixo.

## Reprodutibilidade

Todos os experimentos usam a mesma semente fixa (`SEMENTE = 42`), aplicada tanto na divisão dos dados (treino/validação/teste) quanto na inicialização dos pesos e no embaralhamento dos lotes durante o treinamento, garantindo que os resultados do notebook sejam os mesmos a cada execução.

## Datasets utilizados

- **Regressão:** `load_diabetes` (scikit-learn)
- **Classificação binária:** `load_breast_cancer` (scikit-learn)
- **Classificação multiclasse:** `load_wine` (scikit-learn)

## Uso de IA

Detalhes sobre o uso de ferramentas de IA durante o desenvolvimento estão documentados em [`AI_USAGE.md`](./AI_USAGE.md).
