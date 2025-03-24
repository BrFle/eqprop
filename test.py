from jaxtests import smallest_eigvec
import jax.numpy as jnp
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt


def wrap_jitted(A):
    return smallest_eigvec(A, num_iter=50)


def normalize(v):
    return v / np.linalg.norm(v)


def eigval(A, v):
    return np.dot(v, np.dot(A, v)) / np.dot(v, v)


def calculate(A):
    eigenvector_jax = wrap_jitted(A)

    eigenvector_jax = np.array(eigenvector_jax)

    # eigenvals, eigenvectors_scipy = eigh(A)
    # imin = np.argmin(eigenvals)
    # eigenvector_scipy = eigenvectors_scipy[:, imin]

    # Compare eigenvectors up to sign: the absolute value of their dot
    # product should be 1.
    # dot = np.abs(np.dot(eigenvector_jax, eigenvector_scipy))
    # print(f"eigenval jax: {eigval(A, eigenvector_jax)}")
    # print(f"eigenval scipy: {eigval(A, eigenvector_scipy)}")
    # print("=====================")

    is_really_eigenvector = np.abs(
        np.dot(normalize(np.dot(A, eigenvector_jax)), normalize(eigenvector_jax))
    )

    return is_really_eigenvector
    # return jnp.log(dot)


def test_rayleigh_vs_scipy():
    # Create a random symmetric matrix.
    A_np = np.random.randn(64, 64)
    A_np = (A_np + A_np.T) / 2  # ensure symmetry
    A = jnp.array(A_np)

    return calculate(A)


if __name__ == "__main__":
    np.random.seed(0)
    errors = []
    for i in range(10000):
        errors.append(test_rayleigh_vs_scipy())
    errors = [e for e in errors if jnp.isfinite(e)]
    print(len([e for e in errors if e > -2]))
    plt.hist(errors, bins=50)
    plt.show()
