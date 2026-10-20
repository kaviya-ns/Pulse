import random
import networkx as nx


def generate_ba_graph(n=200, m=3, seed=42):
    return nx.barabasi_albert_graph(n=n, m=m, seed=seed)


def generate_ws_graph(n=200, k=6, p=0.1, seed=42):
    return nx.watts_strogatz_graph(n=n, k=k, p=p, seed=seed)


def generate_bb_graph(n=200, m=3, fitness_fn=None, seed=42):
    rng = random.Random(seed)
    if fitness_fn is None:
        fitness_fn = lambda: rng.uniform(0.1, 1.0)

    G = nx.Graph()
    for i in range(m):
        G.add_node(i, fitness=fitness_fn())
    for i in range(m):
        for j in range(i + 1, m):
            G.add_edge(i, j)

    for new_node in range(m, n):
        eta_new = fitness_fn()
        G.add_node(new_node, fitness=eta_new)

        existing_nodes = list(G.nodes())[:-1]
        weights = [G.nodes[node]['fitness'] * (G.degree[node] + 1e-9)
                   for node in existing_nodes]
        total = sum(weights)
        probs = [w / total for w in weights]

        targets = set()
        while len(targets) < min(m, len(existing_nodes)):
            r = rng.random()
            cumulative = 0.0
            for node, p in zip(existing_nodes, probs):
                cumulative += p
                if r <= cumulative:
                    targets.add(node)
                    break

        for target in targets:
            G.add_edge(new_node, target)

    return G