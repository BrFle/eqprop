import jax
import jax.numpy as jnp
import numpy as np
from sklearn.datasets import make_moons
from tqdm import trange
from tqdm.contrib.concurrent import process_map
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pickle as pkl
from datetime import datetime
from functools import partial
import os
import multiprocessing as mp
import random
import string
import time


def random_id_str():
    return "".join(
        random.Random(os.getpid() + int(time.time())).choices(string.ascii_letters, k=5)
    )


sigmax = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64)
sigmay = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex64)
sigmaz = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64)


def debug_draw(N, n_in, n_out, edges, thetas, ax=None):
    G = nx.Graph()
    for theta, (i, j) in zip(thetas, edges):
        G.add_edge(i, j)
    pos = nx.spring_layout(G)
    node_colors = [
        "green" if node < n_in else "red" if node >= N - n_out else "lightblue"
        for node in G.nodes()
    ]
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=node_colors,
        node_size=500,
        font_weight="bold",
    )
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels)

    # Display the graph
    plt.show()


def debug_matrix(m):
    plt.imshow(np.array(m).real)
    plt.show()


# @jax.jit
def gershgorin_shift(A):
    off_diag_sum = jnp.sum(jnp.abs(A), axis=1) - jnp.abs(jnp.diag(A))
    return jnp.min(jnp.diag(A) - off_diag_sum)


# @partial(jax.jit, static_argnums=(2,))
def smallest_eigvec(A, key=jax.random.PRNGKey(0), num_iter=50):
    # Cf. https://math.iit.edu/~fass/477577_Chapter_10.pdf
    n = A.shape[0]
    sigma = gershgorin_shift(A)
    B = A - sigma * jnp.eye(n, dtype=jnp.complex64)
    # Initialize with a random vector and normalize.
    x = jax.random.normal(key, (n,), dtype=jnp.complex64)
    x = x / jnp.linalg.norm(x)

    def body_fun(_, x):
        y = jnp.linalg.solve(B, x)
        x_new = y / jnp.linalg.norm(y)
        return x_new

    x_final = jax.lax.fori_loop(0, num_iter, body_fun, x)
    return x_final

    # Adaptation to get cubic convergence
    # (but doesn't yield the smallest eigenvalue)
    # mu = jnp.dot(x, jnp.dot(B, x)) - n
    # def body_fun(_, state):
    #     x, mu = state
    #     eps = 1e-6
    #     y = jnp.linalg.solve(B - (mu + eps) * jnp.eye(n, dtype=jnp.complex64), x)
    #     x_new = y / jnp.linalg.norm(y)
    #     mu_new = jnp.dot(x_new, jnp.dot(B, x_new))
    #     return (x_new, mu_new)

    # Use a JAX fori_loop for efficient iteration.
    # x_final, _ = jax.lax.fori_loop(0, num_iter, body_fun, (x, mu))


# @partial(jax.jit, static_argnums=(0, 1))
def promote(N, n, op):
    res = jnp.eye(2, dtype=jnp.complex64)
    if n == 0:
        res = op
    for _ in range(n - 1):
        res = jnp.kron(res, jnp.eye(2, dtype=jnp.complex64))
    if n != 0:
        res = jnp.kron(res, op)
    for _ in range(N - n - 1):
        res = jnp.kron(res, jnp.eye(2, dtype=jnp.complex64))
    return res


# @partial(jax.jit, static_argnums=(0, 1, 2))
def xi_xj(N, i, j, op):
    return promote(N, i, op) @ promote(N, j, op)


# @partial(jax.jit, static_argnums=(0, 1))
def input_matrix(N, edges, uvals):
    total = jnp.zeros((2 ** N, 2 ** N), dtype=jnp.complex64)
    for k, i in enumerate(edges):
        total += promote(N, i, sigmaz) * uvals[k]
    return total


# @partial(jax.jit, static_argnums=(0, 1))
def cost_matrix(N, edges, dvals):
    total = jnp.zeros((2 ** N, 2 ** N), dtype=jnp.complex64)
    for k, i in enumerate(edges):
        D = promote(N, i, sigmaz) - dvals[k] * jnp.eye(2 ** N)
        total += D @ D
    return total


# @partial(jax.jit, static_argnums=(0, 1))
def lattice_matrix(N, edges, thetas):
    total = jnp.eye(2 ** N, dtype=jnp.complex64)
    for k, (i, j) in enumerate(edges):
        total += xi_xj(N, i, j, sigmax) * thetas[k, 0]
        total += xi_xj(N, i, j, sigmaz) * thetas[k, 1]
    return total


# @jax.jit
def dag(x):
    return jnp.conj(x.T)


# @jax.jit
def resum(U, C, L, beta):
    return U + L + (beta / 2) * C


# @partial(jax.jit, static_argnums=(0, 1))
def xixj_expectations(N, edges, psi0):
    res = jnp.zeros(2 * len(edges))
    for k, (i, j) in enumerate(edges):
        res = (
            res.at[2 * k]
            .set(jnp.real(dag(psi0) @ xi_xj(N, i, j, sigmax) @ psi0))
            .at[2 * k + 1]
            .set(jnp.real(dag(psi0) @ xi_xj(N, i, j, sigmaz) @ psi0))
        )
    return res


# @partial(jax.jit, static_argnums=(0, 1))
def compute_gradient(N, edges, U, L, C, beta):
    betanegs = jnp.zeros(2 * len(edges), dtype=jnp.float32)
    betapos = jnp.zeros(2 * len(edges), dtype=jnp.float32)

    H = U + L
    H_tot = H + (beta / 2) * C
    psi0 = smallest_eigvec(H_tot)
    betapos = xixj_expectations(N, edges, psi0)
    H_tot = H - (beta / 2) * C
    psi0 = smallest_eigvec(H_tot)
    betanegs = xixj_expectations(N, edges, psi0)

    return (1 / (2 * beta)) * (betapos - betanegs)


@partial(jax.jit, static_argnums=(0, 1, 2, 3))
def singlestep(N, n_in, n_out, edges, thetas, xy, beta=0.01, rate=0.01):
    u_vertex = range(n_in)
    d_vertex = range(N - n_out, N)
    L = lattice_matrix(N, edges, thetas)
    U = input_matrix(N, u_vertex, xy[:n_in])
    C = cost_matrix(N, d_vertex, xy[n_in:])

    grad = compute_gradient(N, edges, U, L, C, beta)
    return thetas - rate * jnp.reshape(grad, thetas.shape)


@partial(jax.jit, static_argnums=(0, 1, 2, 3))
def adamw_singlestep(
    N,
    n_in,
    n_out,
    edges,
    thetas,
    vs,
    ms,
    k_epoch,
    xy,
    beta=0.01,
    r1=0.9,
    r2=0.999,
    eps=1e-8,
    rate=0.001,
    decay=0.01,
):
    u_vertex = range(n_in)
    d_vertex = range(N - n_out, N)
    L = lattice_matrix(N, edges, thetas)
    U = input_matrix(N, u_vertex, xy[:n_in])
    C = cost_matrix(N, d_vertex, xy[n_in:])

    grad = jnp.reshape(compute_gradient(N, edges, U, L, C, beta), thetas.shape)

    ms = r1 * ms + (1 - r1) * grad
    vs = r2 * vs + (1 - r2) * grad ** 2

    mhat = ms / (1 - r1 ** k_epoch)
    vhat = vs / (1 - r2 ** k_epoch)

    update = rate * mhat / (jnp.sqrt(vhat) + eps)

    thetas = thetas * (1 - rate * decay) - update
    return thetas, vs, ms


@partial(jax.jit, static_argnums=(0, 1, 2))
def calccost(N, n_in, n_out, L, xy):
    u_vertex = range(n_in)
    d_vertex = range(N - n_out, N)
    U = input_matrix(N, u_vertex, xy[:n_in])
    C = cost_matrix(N, d_vertex, xy[n_in:])
    H = U + L
    psi0 = smallest_eigvec(H)
    cost = jnp.real(dag(psi0) @ C @ psi0)

    return cost


@partial(jax.jit, static_argnums=(0, 1, 2, 3))
def predict_all(N, n_in, n_out, edges, thetas, x):
    u_vertex = range(n_in)
    d_vertex = range(N - n_out, N)
    L = lattice_matrix(N, edges, thetas)
    output_ops = [promote(N, i, sigmaz) for i in d_vertex]

    def for_one(x):
        U = input_matrix(N, u_vertex, x)
        H = U + L
        psi0 = smallest_eigvec(H)
        outputs = jnp.empty(n_out, dtype=jnp.complex64)
        for k, op in enumerate(output_ops):
            outputs = outputs.at[k].set(jnp.real(dag(psi0) @ op @ psi0))
        return outputs

    return jax.vmap(for_one)(x)


@partial(jax.jit, static_argnums=(0, 1, 2, 3))
def calcloss_cost(N, n_in, n_out, edges, thetas, testdata):
    L = lattice_matrix(N, edges, thetas)
    costs = jax.vmap(partial(calccost, N, n_in, n_out, L))(testdata)
    return jnp.mean(costs)


def calcloss_result(N, n_in, n_out, edges, thetas, testdata):
    outputs = predict_all(N, n_in, n_out, edges, thetas, testdata[:, :n_in])
    loss = jnp.mean(jnp.abs(jnp.sign(outputs) - testdata[:, n_in:])) / 2
    return loss


def train(N, n_in, n_out, edges, traindata, testdata, thetas, n_epochs=100):
    shuffled = traindata.copy()
    rate = 0.001
    for _ in trange(n_epochs):
        permutation = np.random.permutation(len(shuffled))
        for i in permutation:
            thetas = singlestep(N, n_in, n_out, edges, thetas, traindata[i], rate=rate)
        rate *= 0.99

        yield thetas, calcloss_cost(N, n_in, n_out, edges, thetas, testdata)


def adamw_train(N, n_in, n_out, edges, traindata, testdata, thetas, n_epochs=100):
    shuffled = traindata.copy()
    ms = jnp.zeros_like(thetas)
    vs = jnp.zeros_like(thetas)
    for k in trange(1, n_epochs + 1):
        permutation = np.random.permutation(len(shuffled))
        for i in permutation:
            thetas, vs, ms = adamw_singlestep(
                N, n_in, n_out, edges, thetas, vs, ms, k, traindata[i]
            )

        yield thetas, calcloss_cost(N, n_in, n_out, edges, thetas, testdata)


def generate_spiral_data(samples=100, offset=np.array([0, 0]), classes=[-1, 1]):
    X = np.zeros((samples * len(classes), 3))  # Feature matrix

    for class_number, val in enumerate(classes):
        ix = range(samples * class_number, samples * (class_number + 1))
        r = np.linspace(0.02, 1, samples)  # Radius
        t = (
            np.linspace(0, 2 * np.pi, samples)
            + np.random.randn(samples) * 0.2
            + class_number * 2 * np.pi / len(classes)
        )

        X[ix, 0:2] = np.c_[r * np.sin(t), r * np.cos(t)] + offset
        X[ix, 2] = val

    return X


def generate_moons_data(
    samples=100, noise=0.1, offset=np.array([0, 0]), classes=[-1, 1]
):
    X, y = make_moons(n_samples=samples, noise=noise)
    X += offset
    y = np.array([classes[label] for label in y])
    data = np.hstack((X, y.reshape(-1, 1)))

    return data


def default_colorer(v):
    return "red" if v[0].real > 0 else "blue"


def draw_predictions(
    sc, N, n_in, n_out, edges, thetas, dataset, colorer=default_colorer
):
    predictions = predict_all(N, n_in, n_out, edges, thetas, dataset)
    sc.set_array(predictions[:, 0].real)


def experiment(
    N=6,
    n_in=2,
    n_out=1,
    dataset="spiral",
    scheduler="adamw",
    samples=1000,
    test_samples=100,
    edges=None,
    write_freq=1,
    n_epochs=200,
    **kwargs,
):
    idstr = random_id_str()
    dataset_function = {
        "spiral": generate_spiral_data,
        "moons": generate_moons_data,
    }[dataset]
    train_function = {
        "adamw": adamw_train,
        "vanilla": train,
    }[scheduler]

    if edges is None:
        edges = tuple([(i, j) for i in range(N) for j in range(i + 1, N)])

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    write_dir = f"results/{dataset}_{scheduler}_{idstr}_{timestamp}_{N}"
    os.makedirs(write_dir, exist_ok=True)
    os.chdir(write_dir)

    testdata = jnp.array(dataset_function(samples=test_samples))
    traindata = jnp.array(dataset_function(samples=samples))

    n_epoch = 0
    costs = []
    thetas = jnp.array(np.random.rand(len(edges), 2))

    def write_out():
        outdict = {
            "N": N,
            "n_in": n_in,
            "n_out": n_out,
            "thetas": thetas,
            "edges": edges,
            "costs": costs,
            **kwargs,
        }
        with open(f"costs_{n_epoch}.pkl", "wb") as f:
            pkl.dump(outdict, f)

    for thetas, cost in train_function(
        N, n_in, n_out, edges, traindata, testdata, thetas, n_epochs=n_epochs
    ):
        costs.append(cost)
        if n_epoch % write_freq == 0:
            write_out()
        n_epoch += 1
    write_out()
    os.chdir("..")


def experiment_wrapper(kwargs):
    experiment(**kwargs)


def schedule_experiments(exps):
    process_map(experiment_wrapper, exps, max_workers=7)


def main():
    N = 6
    # edges = tuple([(i, j) for i in range(N) for j in range(i + 1, N)])
    edges = (
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 3),
        (3, 4),
        (4, 2),
        (2, 5),
        (3, 5),
        (4, 5),
    )
    thetas = jnp.array(np.random.rand(len(edges), 2))

    dataset = generate_moons_data
    traindata = jnp.array(dataset(samples=1000))
    testdata = jnp.array(dataset(samples=100))

    costs = []

    # Enable interactive mode
    plt.ion()
    fig, (ax1, ax3, ax2) = plt.subplots(1, 3, figsize=(12, 4))
    (line,) = ax1.plot([], [], "bo-")  # initial empty plot
    debug_draw(N, 2, 1, edges, thetas, ax=ax2)
    sc = ax3.scatter(*testdata[:, :2].T, cmap="coolwarm", c=testdata[:, 2])

    ax1.set_title("Cost")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cost")

    ax2.set_title("Qubit Network")
    kwargs = dict(xdata=[0], ydata=[0], marker="o", markersize=15)
    legend_elements = [
        Line2D(
            markerfacecolor=color,
            label=label,
            **kwargs,
        )
        for color, label in [
            ("green", "Input"),
            ("red", "Output"),
            ("lightblue", "Hidden"),
        ]
    ]
    ax2.legend(handles=legend_elements, loc="upper left")

    ax3.set_title("Predictions")
    fig.tight_layout()

    fig.canvas.draw()  # Update the figure
    fig.show()  # Show the figure
    plt.pause(0.1)  # Pause to allow the plot to update

    # Lists to store points
    xdata, ydata = [], []

    n_epoch = 0

    # Process points from the generator
    for thetas, cost in adamw_train(
        N, 2, 1, edges, traindata, testdata, thetas, n_epochs=1000
    ):
        n_epoch += 1
        x = len(costs)
        y = cost
        costs.append(cost)
        xdata.append(x)
        ydata.append(y)

        # Update the plot data
        line.set_data(xdata, ydata)
        ax1.relim()  # Recompute the data limits
        ax1.autoscale_view()  # Rescale the view to the new data

        if n_epoch % 3 == 0:
            draw_predictions(sc, N, 2, 1, edges, thetas, testdata)

        fig.canvas.draw()  # Update the figure
        fig.show()  # Show the figure
        plt.pause(0.01)  # Pause to allow the plot to update

        if n_epoch % 50 == 0:
            outdict = {
                "N": N,
                "thetas": thetas,
                "edges": edges,
                "costs": costs,
            }
            with open(f"costs_{n_epoch}.pkl", "wb") as f:
                pkl.dump(outdict, f)

    # Optionally keep the plot open after finishing
    plt.ioff()
    breakpoint()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    kwargs = {
        "N": 6,
        "edges": (
            (0, 2),
            (0, 3),
            (0, 4),
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 3),
            (3, 4),
            (4, 2),
            (2, 5),
            (3, 5),
            (4, 5),
        ),
        "dataset": "spiral",
        "n_epochs": 200,
        "write_freq": 5,
    }
    adam_exp = {
        **kwargs,
        "scheduler": "adamw",
    }
    vanilla_exp = {
        **kwargs,
        "scheduler": "vanilla",
    }
    schedule_experiments([adam_exp, vanilla_exp] * 50)
