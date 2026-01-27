import numpy as np
import matplotlib.pyplot as plt

def make_xor_points(n=20, noise=0.06, seed=0):
    rng = np.random.default_rng(seed)

    base = n // 4
    rem = n % 4
    counts = [base + (1 if i < rem else 0) for i in range(4)]

    corners = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])

    Xs = []
    for c, k in zip(corners, counts):
        pts = c + rng.normal(0.0, noise, size=(k, 2))
        pts = np.clip(pts, 0.0, 1.0)
        Xs.append(pts)

    X = np.vstack(Xs)
    rng.shuffle(X)

    xb = (X > 0.5).astype(int)
    y = (xb[:, 0] ^ xb[:, 1]).astype(int)
    return X, y

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def hidden_sigmoid(X):
    x1, x2 = X[:, 0], X[:, 1]
    z1 = 3.0 - 2.0 * x1 - 2.0 * x2
    z2 = -1.0 + 2.0 * x1 + 2.0 * x2
    n1 = sigmoid(z1)
    n2 = sigmoid(z2)
    return n1, n2

def plot_xor_and_hidden(n=20, noise=0.06, seed=0):
    X, y = make_xor_points(n=n, noise=noise, seed=seed)
    n1, n2 = hidden_sigmoid(X)

    c0, c1 = "royalblue", "red"
    colors = np.where(y == 1, c1, c0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=120)

    pad = 0.03

    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], c=colors, s=80, alpha=0.9)
    ax.set_title("1. Input Layer (x1, x2)")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_xlim(-pad, 1 + pad)
    ax.set_ylim(-pad, 1 + pad)
    ax.grid(True)

    ax = axes[1]
    ax.scatter(n1, n2, c=colors, s=80, alpha=0.9)
    ax.set_title("2. Hidden Layer (Sigmoid Result)")
    ax.set_xlabel("n1")
    ax.set_ylabel("n2")
    ax.set_xlim(-pad, 1 + pad)
    ax.set_ylim(-pad, 1 + pad)
    ax.grid(True)

    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], linestyle="--", linewidth=1)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_xor_and_hidden(n=20, noise=0.06, seed=3)

