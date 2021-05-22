from numba import cuda
import numpy
from numpy import *

tamMatriz = 10


@cuda.jit
def my_kernel(matrizA, matrizB, matrizC, width):
    # Thread id in a 1D block
    tx = cuda.blockIdx.x * width + cuda.threadIdx.x

    ty = cuda.blockIdx.y * width + cuda.threadIdx.y

    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x

    i = 0
    soma = 0

    for tx in range(width):
        for ty in range(width):
            soma = 0
            for i in range(width):
                soma = soma + (matrizA[tx][i] * matrizB[i][ty])
            matrizC[tx][ty] = soma


matrizA = random.randint(1, 11, size=(tamMatriz, tamMatriz))
a = numpy.array(matrizA)

matrizB = random.randint(1, 11, size=(tamMatriz, tamMatriz))
b = numpy.array(matrizB)

matrizC = random.randint(0, 1, size=(tamMatriz, tamMatriz))
c = numpy.array(matrizC)

# números de threads por bloco
threads_per_block = 32

# número de blocos por grid
blocks_per_grid = ((len(matrizA) + len(matrizB) // 2) + (threads_per_block - 1))

my_kernel[blocks_per_grid, threads_per_block](a, b, c, tamMatriz)

print(f"Matriz A\n{a}\n")
print(f"Matriz B\n{b}\n")
print(f"\nResultado da Multiplicação das matrizes A*B\n{c}\n")
