import dynamiqs as dq
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from tqdm import trange
from functools import lru_cache


def debug_draw(theta, u, d):
    G = nx.Graph()
    for edge, label in theta.items():
        G.add_edge(edge[0], edge[1], label=str(label))
    pos = nx.spring_layout(G)
    node_colors = [
        "green" if node in u.keys() else "red" if node in d.keys() else "lightblue"
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
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Display the graph
    plt.show()


def debug_matrix(m):
    plt.imshow(m.to_numpy().real)
    plt.show()


def generate_spiral_data(samples=100, classes=3):
    X = np.zeros((samples * classes, 3))  # Feature matrix

    for class_number in range(classes):
        ix = range(samples * class_number, samples * (class_number + 1))
        r = np.linspace(0.0, 1, samples)  # Radius
        t = (
            np.linspace(class_number * 4, (class_number + 1) * 4, samples)
            + np.random.randn(samples) * 0.2
        )  # Theta

        X[ix, 0:2] = np.c_[r * np.sin(t), r * np.cos(t)]
        X[ix, 2] = class_number

    return X


def promote(N, n, op):
    res = dq.eye(2)
    if n == 0:
        res = op
    for _ in range(n - 1):
        res = dq.tensor(res, dq.eye(2))
    if n != 0:
        res = dq.tensor(res, op)
    for _ in range(N - n - 1):
        res = dq.tensor(res, dq.eye(2))
    return res


def _u(a):
    return a[0][0]


class QuantumEqPropagator:
    def __init__(self, N, beta, thetas, us, ds):
        self.N = N
        self.beta = beta
        self.thetas = thetas
        self.us = us
        self.ds = ds
        self.U = self._generate_input(us)
        self.C = self._generate_cost(ds)
        self.L = self._generate_lattice(thetas)

        self.H = self.update_hamiltonian(beta)

    def _generate_cost(self, ds):
        to_add = []
        for i, d in ds.items():
            D = promote(self.N, i, dq.sigmaz()) - d * dq.eye(*[2] * self.N)
            to_add.append(D @ D)
        return dq.stack(to_add).sum(axis=0)

    @lru_cache
    def _XiXj(self, i, j, op):
        return promote(self.N, i, op()) @ promote(self.N, j, op())

    def _generate_lattice(self, thetas):
        to_add = []
        for (i, j), (a, b) in thetas.items():
            to_add.append(self._XiXj(i, j, dq.sigmax) * a)
            to_add.append(self._XiXj(i, j, dq.sigmaz) * b)
        return dq.stack(to_add).sum(axis=0)

    def _generate_input(self, us):
        to_add = []
        for i, u in us.items():
            to_add.append(promote(self.N, i, dq.sigmaz()) * u)
        return dq.stack(to_add).sum(axis=0)

    def update_hamiltonian(self, beta=None):
        to_add = [self.U, self.L]
        if beta is not None:
            to_add.append(self.C * beta / 2)
        return dq.stack(to_add).sum(axis=0)

    def set_input(self, us, partial=True):
        self.us = us
        self.U = self._generate_input(us)
        if not partial:
            self.H = self.update_hamiltonian(self.beta)

    def set_cost(self, ds, partial=True):
        self.ds = ds
        self.C = self._generate_cost(ds)
        if not partial:
            self.H = self.update_hamiltonian(self.beta)

    def set_lattice(self, thetas, partial=True):
        self.thetas = thetas
        self.L = self._generate_lattice(thetas)
        if not partial:
            self.H = self.update_hamiltonian(self.beta)

    def set_beta(self, beta):
        self.beta = beta
        self.H = self.update_hamiltonian(beta)

    def ground_state(self):
        H = self.H.to_numpy()
        eigE, psi0 = eigsh(H, k=1, which="SA")
        return dq.asqarray(psi0, dims=(2,) * self.N)

    def compute_gradient(self, beta):
        betanegs = np.zeros(2 * len(self.thetas))
        betapos = np.zeros(2 * len(self.thetas))

        for vals, b in ((betanegs, -beta), (betapos, beta)):
            self.set_beta(b)
            gs = self.ground_state()
            for k, (i, j) in enumerate(self.thetas.keys()):
                vals[2 * k] = float(dq.expect(self._XiXj(i, j, dq.sigmax), gs).real)
                vals[2 * k + 1] = float(dq.expect(self._XiXj(i, j, dq.sigmaz), gs).real)

        grad = 1 / (2 * beta) * (betapos - betanegs)
        return grad

    def train(self, data, rate, beta, n_epochs=1):
        shuffled = data.copy()
        costs = np.empty(len(data))
        for _ in trange(n_epochs):
            np.random.shuffle(shuffled)
            for i in trange(len(shuffled)):
                x = shuffled[i, 0:2]
                y = shuffled[i, 2]
                self.set_input({0: x[0], 1: x[1]})
                self.set_cost({4: 1 if y == 0 else -1, 5: 1 if y == 1 else -1})
                grad = self.compute_gradient(beta)
                for k, (i, j) in enumerate(self.thetas.keys()):
                    self.thetas[(i, j)] = (
                        self.thetas[(i, j)][0] - rate * grad[2 * k],
                        self.thetas[(i, j)][1] - rate * grad[2 * k + 1],
                    )
                self.set_lattice(self.thetas)

                psi0 = self.ground_state()
                costs[i] = float(dq.expect(self.C, psi0).real)

            self.set_beta(0)
            yield np.average(costs)


N = 6
beta = 1
us = {0: 1, 1: 1}
thetas = {
    (0, 2): (1, 1),
    (0, 3): (1, 1),
    (1, 2): (1, 1),
    (1, 3): (1, 1),
    (2, 4): (1, 1),
    (2, 5): (1, 1),
    (3, 4): (1, 1),
    (3, 5): (1, 1),
}
ds = {4: 1, 5: 1}

h = QuantumEqPropagator(N, beta, thetas, us, ds)

data = generate_spiral_data(samples=10, classes=2)

costs = []
for cost in h.train(data, 0.3, 0.1, n_epochs=3):
    costs.append(cost)

print(costs)
