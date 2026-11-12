import numpy as np
from mlp.base import Ativacao


def _sigmoid(entradas: np.ndarray) -> np.ndarray:
	saida = np.empty_like(entradas, dtype=float)
	positivas = entradas >= 0
	saida[positivas] = 1.0 / (1.0 + np.exp(-entradas[positivas]))
	exponencial = np.exp(entradas[~positivas])
	saida[~positivas] = exponencial / (1.0 + exponencial)
	return saida


class Sigmoid(Ativacao):
    def passo_a_frente(self, entradas: np.ndarray) -> np.ndarray:
		self.entradas = np.array(entradas, copy=True)
		self.saida = _sigmoid(self.entradas)
		return self.saida

	def passo_atras(self, gradiente_saida: np.ndarray) -> np.ndarray:
		return gradiente_saida * self.saida * (1.0 - self.saida)


class ReLU(Ativacao):
	def passo_a_frente(self, entradas: np.ndarray) -> np.ndarray:
		self.entradas = np.array(entradas, copy=True)
		return np.maximum(self.entradas, 0.0)

	def passo_atras(self, gradiente_saida: np.ndarray) -> np.ndarray:
		return gradiente_saida * (self.entradas > 0.0)


class LeakyReLU(Ativacao):
    def __init__(self, alpha: float = 0.01):
		self.alpha = alpha

	def passo_a_frente(self, entradas: np.ndarray) -> np.ndarray:
		self.entradas = np.array(entradas, copy=True)
		return np.where(self.entradas > 0.0, self.entradas, self.alpha * self.entradas)

	def passo_atras(self, gradiente_saida: np.ndarray) -> np.ndarray:
		derivada = np.where(self.entradas > 0.0, 1.0, self.alpha)
		return gradiente_saida * derivada


class ELU(Ativacao):
	def __init__(self, alpha: float = 1.0):
		self.alpha = alpha

	def passo_a_frente(self, entradas: np.ndarray) -> np.ndarray:
		self.entradas = np.array(entradas, copy=True)
		saida = self.entradas.copy()
		negativas = self.entradas <= 0.0
		saida[negativas] = self.alpha * np.expm1(self.entradas[negativas])
		return saida

	def passo_atras(self, gradiente_saida: np.ndarray) -> np.ndarray:
		derivada = np.ones_like(self.entradas, dtype=float)
		negativas = self.entradas <= 0.0
		derivada[negativas] = self.alpha * np.exp(self.entradas[negativas])
		return gradiente_saida * derivada


class Softplus(Ativacao):
	def passo_a_frente(self, entradas: np.ndarray) -> np.ndarray:
		self.entradas = np.array(entradas, copy=True)
		return np.logaddexp(0.0, self.entradas)

	def passo_atras(self, gradiente_saida: np.ndarray) -> np.ndarray:
		return gradiente_saida * _sigmoid(self.entradas)


class Softsign(Ativacao):
	def passo_a_frente(self, entradas: np.ndarray) -> np.ndarray:
		self.entradas = np.array(entradas, copy=True)
		return self.entradas / (1.0 + np.abs(self.entradas))

	def passo_atras(self, gradiente_saida: np.ndarray) -> np.ndarray:
		derivada = 1.0 / (1.0 + np.abs(self.entradas)) ** 2
		return gradiente_saida * derivada


class Swish(Ativacao):
	def passo_a_frente(self, entradas: np.ndarray) -> np.ndarray:
		self.entradas = np.array(entradas, copy=True)
		self.saida_sigmoid = _sigmoid(self.entradas)
		return self.entradas * self.saida_sigmoid

	def passo_atras(self, gradiente_saida: np.ndarray) -> np.ndarray:
		derivada = self.saida_sigmoid + self.entradas * self.saida_sigmoid * (
			1.0 - self.saida_sigmoid
		)
		return gradiente_saida * derivada


class HardSigmoid(Ativacao):
	def passo_a_frente(self, entradas: np.ndarray) -> np.ndarray:
		self.entradas = np.array(entradas, copy=True)
		return np.clip(0.2 * self.entradas + 0.5, 0.0, 1.0)

	def passo_atras(self, gradiente_saida: np.ndarray) -> np.ndarray:
		derivada = 0.2 * (np.abs(self.entradas) < 2.5)
		return gradiente_saida * derivada


class Softmax(Ativacao):
	def passo_a_frente(self, entradas: np.ndarray) -> np.ndarray:
		self.entradas = np.array(entradas, copy=True)
		deslocadas = self.entradas - np.max(self.entradas, axis=1, keepdims=True)
		exponenciais = np.exp(deslocadas)
		self.saida = exponenciais / np.sum(exponenciais, axis=1, keepdims=True)
		return self.saida

	def passo_atras(self, gradiente_saida: np.ndarray) -> np.ndarray:
		produto_soma = np.sum(
			gradiente_saida * self.saida, axis=1, keepdims=True
		)
		return self.saida * (gradiente_saida - produto_soma)


class Linear(Ativacao):
	def passo_a_frente(self, entradas: np.ndarray) -> np.ndarray:
		self.entradas = np.array(entradas, copy=True)
		return self.entradas.copy()

	def passo_atras(self, gradiente_saida: np.ndarray) -> np.ndarray:
		return gradiente_saida