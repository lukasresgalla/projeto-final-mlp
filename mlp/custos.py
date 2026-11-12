import numpy as np
from mlp.base import FuncaoCusto


class MSE(FuncaoCusto):
	def calcular_custo(
		self, y_verdadeiro: np.ndarray, y_predito: np.ndarray
	) -> float:
		return float(np.mean((y_predito - y_verdadeiro) ** 2))

	def gradiente(
		self, y_verdadeiro: np.ndarray, y_predito: np.ndarray
	) -> np.ndarray:
		return 2.0 * (y_predito - y_verdadeiro) / y_verdadeiro.size


class EntropiaCruzadaBinaria(FuncaoCusto):
	def calcular_custo(
		self, y_verdadeiro: np.ndarray, y_predito: np.ndarray
	) -> float:
		p = np.clip(y_predito, 1e-12, 1.0 - 1e-12)
		custo = -np.mean(
			y_verdadeiro * np.log(p)
			+ (1.0 - y_verdadeiro) * np.log(1.0 - p)
		)
		return float(custo)

	def gradiente(
		self, y_verdadeiro: np.ndarray, y_predito: np.ndarray
	) -> np.ndarray:
		p = np.clip(y_predito, 1e-12, 1.0 - 1e-12)
		m = y_verdadeiro.shape[0]
		return (p - y_verdadeiro) / (p * (1.0 - p)) / m


class EntropiaCruzadaCategorica(FuncaoCusto):
	def calcular_custo(
		self, y_verdadeiro: np.ndarray, y_predito: np.ndarray
	) -> float:
		p = np.clip(y_predito, 1e-12, 1.0 - 1e-12)
		return float(-np.sum(y_verdadeiro * np.log(p)) / y_verdadeiro.shape[0])

	def gradiente(
		self, y_verdadeiro: np.ndarray, y_predito: np.ndarray
	) -> np.ndarray:
		p = np.clip(y_predito, 1e-12, 1.0 - 1e-12)
		m = y_verdadeiro.shape[0]
		return -y_verdadeiro / p / m