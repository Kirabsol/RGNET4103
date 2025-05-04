import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import itertools
from math import log


output_dir = "/home/ubuntu/Results_Q4"
os.makedirs(output_dir, exist_ok=True)


def common_neighbors(G, u, v):
    """Compute the number of common neighbors between nodes u and v."""
    u_neighbors = set(G.neighbors(u))
    v_neighbors = set(G.neighbors(v))
    return len(u_neighbors.intersection(v_neighbors))

def jaccard_coefficient(G, u, v):
    """Compute the Jaccard coefficient between nodes u and v."""
    u_neighbors = set(G.neighbors(u))
    v_neighbors = set(G.neighbors(v))
    intersection_len = len(u_neighbors.intersection(v_neighbors))
    union_len = len(u_neighbors.union(v_neighbors))
    if union_len == 0:
        return 0.0
    return intersection_len / union_len

def adamic_adar_index(G, u, v):
    """Compute the Adamic/Adar index between nodes u and v."""
    score = 0.0
    u_neighbors = set(G.neighbors(u))
    v_neighbors = set(G.neighbors(v))
    common = u_neighbors.intersection(v_neighbors)
    for node in common:
        degree_node = G.degree(node)
        if degree_node > 1:
            score += 1 / log(degree_node)
    return score


def split_edges(G, test_fraction=0.1):
    """Randomly remove a fraction of edges for testing."""
    edges = list(G.edges())
    n_edges = len(edges)
    n_test_edges = int(n_edges * test_fraction)
    
    random.shuffle(edges)
    
    test_edges = edges[:n_test_edges]
    train_edges = edges[n_test_edges:]
    
    G_train = nx.Graph()
    G_train.add_nodes_from(G.nodes())
    G_train.add_edges_from(train_edges)
    
    
    return G_train, set(test_edges), set(train_edges)

def get_candidate_edges(G_train):
    """Generate candidate edges (pairs of non-connected nodes)."""
    candidate_edges = set()
    nodes = list(G_train.nodes())
    for u, v in itertools.combinations(nodes, 2):
        if not G_train.has_edge(u, v):
            candidate_edges.add(tuple(sorted((u, v))))
    return candidate_edges

def predict_links(G_train, candidate_edges, predictor_func):
    """Compute prediction scores for candidate edges."""
    predictions = []
    for u, v in candidate_edges:
        score = predictor_func(G_train, u, v)
        predictions.append(((u, v), score))
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions

def evaluate_predictions(predictions, test_edges, k_values):
    """Compute precision@k and recall@k."""
    results = {k: {"precision": 0.0, "recall": 0.0} for k in k_values}
    predicted_edges_at_k = {k: set() for k in k_values}
    
    test_edges_set = set(map(lambda x: tuple(sorted(x)), test_edges)) 
    num_test_edges = len(test_edges_set)
    if num_test_edges == 0:
        print("Warning: Number of test edges is 0. Recall will be 0.")
        return results

    num_correct_at_k = {k: 0 for k in k_values}
    
    for i, (edge, score) in enumerate(predictions):
        edge_sorted = tuple(sorted(edge))
        is_correct = edge_sorted in test_edges_set
        
        for k in k_values:
            if i < k:
                predicted_edges_at_k[k].add(edge_sorted)
                if is_correct:
                    num_correct_at_k[k] += 1
            
    for k in k_values:
        precision = num_correct_at_k[k] / k if k > 0 else 0.0
        recall = num_correct_at_k[k] / num_test_edges if num_test_edges > 0 else 0.0
        results[k]["precision"] = precision
        results[k]["recall"] = recall
        
    return results



files = os.listdir("/home/ubuntu/NET4103/fb100/data")

fraction_values = [0.05, 0.1, 0.15, 0.2]
k_values = list(range(50, 401, 50))
predictors = {
    "Common Neighbors": common_neighbors,
    "Jaccard": jaccard_coefficient,
    "Adamic/Adar": adamic_adar_index
}

all_results = {}

for name in files:
    print(f"\n--- Processing Graph: {name} ---")
    all_results[name] = {}
    try:
        G_full = nx.read_gml(f"/home/ubuntu/NET4103/fb100/data/{name}", label='id')
        if G_full.is_directed():
            G_full = G_full.to_undirected()
        connected_components = sorted(nx.connected_components(G_full), key=len, reverse=True)
        if not connected_components:
            print(f"  Error: No connected components found. Skipping.")
            continue
        G = G_full.subgraph(connected_components[0]).copy()
        G.remove_nodes_from(list(nx.isolates(G)))
        print(f"  Using LCC with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

        all_results[name]["metrics"] = {f: {p_name: {} for p_name in predictors} for f in fraction_values}

        for f in fraction_values:
            print(f"\n  Testing with fraction f = {f}")
            G_train, test_edges, _ = split_edges(G, test_fraction=f)
            print(f"    Train graph: {G_train.number_of_nodes()} nodes, {G_train.number_of_edges()} edges")
            print(f"    Test edges: {len(test_edges)}")

            if not test_edges:
                print("    Skipping evaluation for f={f} due to 0 test edges.")
                continue

            print("    Generating candidate edges...")
            candidate_edges = get_candidate_edges(G_train)
            print(f"    Number of candidate edges: {len(candidate_edges)}")

            if not candidate_edges:
                print("    Skipping evaluation for f={f} due to 0 candidate edges.")
                continue

            for p_name, predictor_func in predictors.items():
                print(f"\n    Evaluating predictor: {p_name}")
                
                print("      Predicting links...")
                predictions = predict_links(G_train, candidate_edges, predictor_func)
                
                print("      Evaluating predictions...")
                eval_results = evaluate_predictions(predictions, test_edges, k_values)
                all_results[name]["metrics"][f][p_name] = eval_results
                
                for k in k_values:
                    prec = eval_results[k]['precision']
                    rec = eval_results[k]['recall']
                    print(f"        k={k}: Precision={prec:.4f}, Recall={rec:.4f}")

    except Exception as e:
        print(f"  Error processing graph {name}: {e}")


print("\nGenerating example plots (Precision@k for f=0.1)...")
plot_f = 0.1 
for name in files:
    if name not in all_results or plot_f not in all_results[name].get("metrics", {}):
        print(f"  Skipping plot for {name} (no results for f={plot_f})")
        continue

    fig, ax = plt.subplots(figsize=(10, 6))
    for p_name in predictors.keys():
        if p_name not in all_results[name]["metrics"][plot_f]:
            continue
        
        precision_values = [all_results[name]["metrics"][plot_f][p_name][k]['precision'] for k in k_values]
        ax.plot(k_values, precision_values, marker='o', linestyle='-', label=p_name)

    ax.set_title(f"Link Prediction Precision@k for {name} (f={plot_f})")
    ax.set_xlabel("k (Number of Top Predictions)")
    ax.set_ylabel("Precision@k")
    ax.set_xticks(k_values)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    plot_path = os.path.join(output_dir, f"{name}_precision_at_k_f{plot_f}.png")
    try:
        fig.savefig(plot_path)
        print(f"  Saved plot to {plot_path}")
    except Exception as plot_err:
        print(f"  Error saving plot for {name}: {plot_err}")
    plt.close(fig)



