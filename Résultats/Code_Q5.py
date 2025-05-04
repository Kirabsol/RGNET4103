import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import os
from collections import Counter
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import warnings

output_dir = "/home/ubuntu/Results_Q5"
os.makedirs(output_dir, exist_ok=True)

# --- Label Propagation Algorithm Implementation (Iterative Version) ---

def label_propagation_predict(G, attribute_name, known_labels, max_iter=100, tol=1e-3):

    current_labels = known_labels.copy()
    unknown_nodes = [node for node in G.nodes() if node not in known_labels]
    for node in unknown_nodes:
        current_labels[node] = -1

    nodes = list(G.nodes())
    changed = True
    iter_count = 0

    while changed and iter_count < max_iter:
        changed = False
        iter_count += 1
        print(f"  Iteration {iter_count}/{max_iter}...")
        
        random.shuffle(nodes)
        
        new_labels = current_labels.copy()
        
        for node in nodes:
            if node in unknown_nodes:
                neighbors = list(G.neighbors(node))
                if not neighbors:
                    continue

                neighbor_labels = [current_labels[neighbor] for neighbor in neighbors if current_labels[neighbor] != -1]
                
                if not neighbor_labels:
                    continue

                label_counts = Counter(neighbor_labels)
                
                max_count = label_counts.most_common(1)[0][1]
                most_frequent_labels = [label for label, count in label_counts.items() if count == max_count]
                
                new_label = random.choice(most_frequent_labels)
                
                if new_labels[node] != new_label:
                    new_labels[node] = new_label
                    changed = True

        current_labels = new_labels
        

    print(f"  Label propagation finished after {iter_count} iterations.")
    
    final_labels = current_labels.copy()
    remaining_unknown = [node for node, label in final_labels.items() if label == -1]
    if remaining_unknown:
        print(f"  Warning: {len(remaining_unknown)} nodes still have unknown labels after max iterations.")

        known_values = [label for node, label in known_labels.items()]
        if known_values:
            overall_most_frequent = Counter(known_values).most_common(1)[0][0]
            for node in remaining_unknown:
                final_labels[node] = overall_most_frequent
        else:
             for node in remaining_unknown:
                 final_labels[node] = 0

    return final_labels


def evaluate_recovery(true_labels, predicted_labels, nodes_to_evaluate):
    """
    Calculates accuracy and MAE for the recovered labels.
    """
    true_list = []
    pred_list = []
    
    for node in nodes_to_evaluate:
        if node in true_labels and node in predicted_labels:
            true_list.append(true_labels[node])
            pred_list.append(predicted_labels[node])
        else:
            print(f"Warning: Node {node} missing from true or predicted labels during evaluation.")

    if not true_list: 
        return 0.0, 0.0

    try:
        true_numeric = np.array(true_list, dtype=float)
        pred_numeric = np.array(pred_list, dtype=float)
        mae = mean_absolute_error(true_numeric, pred_numeric)
    except ValueError:
        print("Warning: Could not convert labels to numeric for MAE calculation. Setting MAE to NaN.")
        mae = np.nan
        
    accuracy = accuracy_score(true_list, pred_list)
    
    return accuracy, mae



network_name = "MIT8"
file_path = "/home/ubuntu/NET4103/fb100/data/MIT8.gml"
attributes = ["dorm", "major_index", "gender"]
removal_percentages = [0.1, 0.2, 0.3]
max_iterations_lpa = 50

results = {attr: {p: {"accuracy": 0.0, "mae": 0.0} for p in removal_percentages} for attr in attributes}

print(f"--- Processing Graph: {network_name} for Label Propagation ---")
try:
    G_full = nx.read_gml(file_path, label='id')

    if G_full.is_directed():
        G_full = G_full.to_undirected()
    connected_components = sorted(nx.connected_components(G_full), key=len, reverse=True)
    if not connected_components:
        raise ValueError("Graph has no connected components.")
    G = G_full.subgraph(connected_components[0]).copy()
    G.remove_nodes_from(list(nx.isolates(G)))
    print(f"Using LCC with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    for attribute in attributes:
        print(f"\n-- Recovering Attribute: {attribute} --")
        
        true_labels_all = nx.get_node_attributes(G, attribute)
        
        if not true_labels_all:
            print(f"Attribute '{attribute}' not found in graph. Skipping.")
            continue
            
        nodes_with_attribute = list(true_labels_all.keys())
        num_nodes_total = len(nodes_with_attribute)
        if num_nodes_total == 0:
             print(f"No nodes have attribute '{attribute}'. Skipping.")
             continue

        unique_labels = sorted(list(set(true_labels_all.values())))
        label_map = {label: i for i, label in enumerate(unique_labels)}
        true_labels_int = {node: label_map[label] for node, label in true_labels_all.items()}

        for p in removal_percentages:
            print(f"\nTesting with {p*100:.0f}% missing labels")
            
            num_to_remove = int(num_nodes_total * p)
            nodes_to_hide = random.sample(nodes_with_attribute, num_to_remove)
            
            known_labels = {node: label for node, label in true_labels_int.items() if node not in nodes_to_hide}
            nodes_to_evaluate = nodes_to_hide
            
            print(f"Number of known labels: {len(known_labels)}")
            print(f"Number of labels to predict: {len(nodes_to_evaluate)}")


            predicted_labels = label_propagation_predict(G, attribute, known_labels, max_iter=max_iterations_lpa)
            
            accuracy, mae = evaluate_recovery(true_labels_int, predicted_labels, nodes_to_evaluate)
            results[attribute][p]["accuracy"] = accuracy
            results[attribute][p]["mae"] = mae
            
            print(f"Evaluation Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"MAE: {mae:.4f}" if not np.isnan(mae) else "MAE: N/A")

except Exception as e:
    print(f"Error processing graph {network_name} for Q5: {e}")




