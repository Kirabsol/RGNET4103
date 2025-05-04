import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain
import numpy as np
import os
import random
from collections import Counter
from sklearn.metrics.cluster import normalized_mutual_info_score
import warnings


output_dir = "/home/ubuntu/Results_Q6"
os.makedirs(output_dir, exist_ok=True)

files = os.listdir("/home/ubuntu/NET4103/fb100/data")

def plot_communities(G, partition, title, filename):
    try:
        pos = nx.spring_layout(G, seed=42)
        unique_communities = sorted(list(set(partition.values())))
        num_unique_communities = len(unique_communities)
        community_map = {comm_id: i for i, comm_id in enumerate(unique_communities)}
        node_colors = [community_map[partition[node]] for node in G.nodes()]
        
        cmap = plt.cm.get_cmap("viridis", num_unique_communities)
        
        plt.figure(figsize=(12, 12))
        nx.draw_networkx_nodes(G, pos, node_size=40,
                               cmap=cmap, node_color=node_colors, alpha=0.8)
        nx.draw_networkx_edges(G, pos, alpha=0.1)
        plt.title(title)
        plt.axis("off")
        plt.savefig(filename)
        plt.close()
        print(f"Saved community visualization to {filename}")
    except Exception as e:
        print(f"Error generating community plot for {title}: {e}")

# --- Analysis Loop ---
for name, file_path in graph_files_to_analyze.items():
    print(f"\n--- Processing Graph: {name} ---")
    try:
        G_full = nx.read_gml(file_path, label='id')
        
        if G_full.is_directed():
            G_full = G_full.to_undirected()
            
        connected_components = sorted(nx.connected_components(G_full), key=len, reverse=True)
        if not connected_components:
            print(f"Error: No connected components found. Skipping.")
            continue
            
        G = G_full.subgraph(connected_components[0]).copy()
        G.remove_nodes_from(list(nx.isolates(G)))
        print(f"Using LCC with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

        print("Detecting communities using Louvain algorithm...")
        partition = community_louvain.best_partition(G, weight=None, random_state=42)
        num_communities = max(partition.values()) + 1
        modularity = community_louvain.modularity(partition, G, weight=None)
        
        print(f"Detected {num_communities} communities.")
        print(f"Modularity: {modularity:.4f}")


        print("Validating hypothesis by comparing communities with attributes...")
        

        nodes_in_partition = list(partition.keys())
        nodes_in_graph = list(G.nodes())
        if set(nodes_in_partition) != set(nodes_in_graph):
             print("Warning: Nodes in partition do not match nodes in graph LCC. Re-aligning.")
             aligned_partition = {node: partition[node] for node in nodes_in_graph if node in partition}
             nodes_for_eval = [node for node in nodes_in_graph if node in aligned_partition]
        else:
             aligned_partition = partition
             nodes_for_eval = nodes_in_graph
             
        detected_communities_list = [aligned_partition[node] for node in nodes_for_eval]
        
        results_nmi = {}
        for attr in ["dorm", "major_index"]:
            true_labels_attr = nx.get_node_attributes(G, attr)
            if true_labels_attr:
                attribute_list = [true_labels_attr.get(node, -1) for node in nodes_for_eval]
                
                valid_indices = [i for i, label in enumerate(attribute_list) if label != -1]
                if len(valid_indices) < len(nodes_for_eval):
                     print(f"Warning: Attribute '{attr}' missing for some nodes. Evaluating on subset.")
                
                filtered_detected = [detected_communities_list[i] for i in valid_indices]
                filtered_attribute = [attribute_list[i] for i in valid_indices]
                
                if len(set(filtered_attribute)) > 1 and len(set(filtered_detected)) > 1:
                    try:
                        nmi = normalized_mutual_info_score(filtered_attribute, filtered_detected, average_method='arithmetic')
                        results_nmi[attr] = nmi
                        print(f"NMI (Communities vs {attr}): {nmi:.4f}")
                    except Exception as nmi_err:
                        print(f"Error calculating NMI for {attr}: {nmi_err}")
                else:
                    print(f"kipping NMI for {attr}: Not enough unique labels in attribute or detected communities after filtering.")
            else:
                print(f"Attribute '{attr}' not found. Cannot calculate NMI.")

        nmi_dorm = results_nmi.get("dorm", -1)
        nmi_major = results_nmi.get("major_index", -1)
        
        conclusion = "Conclusion: "
        if nmi_dorm == -1 and nmi_major == -1:
            conclusion += "Could not calculate NMI for either dorm or major."
        elif nmi_dorm == -1:
            conclusion += f"Could not calculate NMI for dorm. NMI for major is {nmi_major:.4f}. Cannot compare."
        elif nmi_major == -1:
             conclusion += f"Could not calculate NMI for major. NMI for dorm is {nmi_dorm:.4f}. Cannot compare."
        elif abs(nmi_dorm - nmi_major) < 0.01: # Consider them similar if difference is small
             conclusion += f"The NMI values for dorm ({nmi_dorm:.4f}) and major ({nmi_major:.4f}) are very similar, suggesting both play a comparable role or neither strongly dominates community structure as detected."
        elif nmi_dorm > nmi_major:
            conclusion += f"The NMI for dorm ({nmi_dorm:.4f}) is higher than for major ({nmi_major:.4f}), supporting the hypothesis that dorm assignments align more strongly with detected community structures in this network."
        else: # nmi_major > nmi_dorm
            conclusion += f"The NMI for major ({nmi_major:.4f}) is higher than for dorm ({nmi_dorm:.4f}), contradicting the hypothesis. Major seems to be a stronger factor in community formation here."
        
        print(f"  {conclusion}")

    except Exception as e:
        print(f"  Error processing graph {name}: {e}")



