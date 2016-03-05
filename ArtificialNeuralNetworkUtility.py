import random

def between(min, max):
	"""
	Get a random value from the given interval
	"""
	return random.random() * (max - min) + min

def make_matrix(N, M):
	"""
	Make a matrix (N x M)
	"""
	return [[0 for i in range(M)] for i in range(N)]