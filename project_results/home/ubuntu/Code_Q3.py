import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
from collections import Counter


output_dir = "/home/ubuntu/Results_Q3"
os.makedirs(output_dir, exist_ok=True)

files = os.listdir("/home/ubuntu/NET4103/fb100/data")

attributes_to_analyze = {
    "student_fac": "Student/Faculty Status",
    "major_index": "Major",
    "dorm": "Dorm",
    "gender": "Gender"
}

results = {
    "network_size": [],
    "student_fac": [],
    "major_index": [],
    "degree": [],
    "dorm": [],
    "gender": []
}
network_names_processed = []


for name in files:
    print(f"Processing {name} network...")
    try:
        G_full = nx.read_gml(f"/home/ubuntu/NET4103/fb100/data/{name}", label='id') 
        print(f"{name} (Full): Nodes={G_full.number_of_nodes()}, Edges={G_full.number_of_edges()}")

        node_list = list(G_full.nodes())
        if node_list and isinstance(node_list[0], str):
             try:
                 G_full = nx.convert_node_labels_to_integers(G_full, first_label=0, ordering='default')
                 print("Converted node labels to integers.")
             except Exception as conv_error:
                 print(f"Warning: Could not convert node labels to integers for {name}: {conv_error}. Trying to proceed.")

        if G_full.is_directed():
            G_full = G_full.to_undirected()
            print("Converted graph to undirected.")

        connected_components = sorted(nx.connected_components(G_full), key=len, reverse=True)
        if not connected_components:
             print(f"Error: No connected components found for {name}. Skipping.")
             continue
        lcc_nodes = connected_components[0]
        G = G_full.subgraph(lcc_nodes).copy() #
        
        G.remove_nodes_from(list(nx.isolates(G)))
        
        n_nodes = G.number_of_nodes()
        if n_nodes == 0:
            print(f"Error: LCC has 0 nodes after processing for {name}. Skipping.")
            continue
        print(f"{name} (LCC): Nodes={n_nodes}, Edges={G.number_of_edges()}")

        results["network_size"].append(n_nodes)
        network_names_processed.append(name)


        attr = "student_fac"
        if nx.get_node_attributes(G, attr):
            attr_values = nx.get_node_attributes(G, attr)
            if len(set(attr_values.values())) > 1: 
                try:
                    assortativity = nx.attribute_assortativity_coefficient(G, attr)
                    results[attr].append(assortativity)
                    print(f"{attributes_to_analyze[attr]}: {assortativity:.4f}")
                except Exception as e:
                    print(f"Error calculating assortativity for {attr}: {e}")
                    results[attr].append(np.nan) 
            else:
                print(f"Attribute '{attr}' has only one unique value. Assortativity undefined.")
                results[attr].append(np.nan)
        else:
            print(f"Attribute '{attr}' not found or empty.")
            results[attr].append(np.nan)


        attr = "major_index"
        if nx.get_node_attributes(G, attr):
            attr_values = nx.get_node_attributes(G, attr)
            if len(set(attr_values.values())) > 1:
                try:
                    assortativity = nx.attribute_assortativity_coefficient(G, attr)
                    results[attr].append(assortativity)
                    print(f"{attributes_to_analyze[attr]}: {assortativity:.4f}")
                except Exception as e:
                    print(f"Error calculating assortativity for {attr}: {e}")
                    results[attr].append(np.nan)
            else:
                print(f"Attribute '{attr}' has only one unique value. Assortativity undefined.")
                results[attr].append(np.nan)
        else:
            print(f"Attribute '{attr}' not found or empty.")
            results[attr].append(np.nan)


        if G.number_of_edges() > 0:
            try:
                assortativity = nx.degree_assortativity_coefficient(G)
                results["degree"].append(assortativity)
                print(f"Degree: {assortativity:.4f}")
            except Exception as e:
                print(f"Error calculating degree assortativity: {e}")
                results["degree"].append(np.nan)
        else:
             print(f"Graph has no edges. Degree assortativity undefined.")
             results["degree"].append(np.nan)


        attr = "dorm"
        if nx.get_node_attributes(G, attr):
            attr_values = nx.get_node_attributes(G, attr)
            if len(set(attr_values.values())) > 1:
                try:
                    assortativity = nx.attribute_assortativity_coefficient(G, attr)
                    results[attr].append(assortativity)
                    print(f"{attributes_to_analyze[attr]}: {assortativity:.4f}")
                except Exception as e:
                    print(f"Error calculating assortativity for {attr}: {e}")
                    results[attr].append(np.nan)
            else:
                print(f"Attribute '{attr}' has only one unique value. Assortativity undefined.")
                results[attr].append(np.nan)
        else:
            print(f"Attribute '{attr}' not found or empty.")
            results[attr].append(np.nan)


        attr = "gender"
        if nx.get_node_attributes(G, attr):
            attr_values = nx.get_node_attributes(G, attr)
            if len(set(attr_values.values())) > 1:
                try:
                    assortativity = nx.attribute_assortativity_coefficient(G, attr)
                    results[attr].append(assortativity)
                    print(f"{attributes_to_analyze[attr]}: {assortativity:.4f}")
                except Exception as e:
                    print(f"Error calculating assortativity for {attr}: {e}")
                    results[attr].append(np.nan)
            else:
                print(f"Attribute '{attr}' has only one unique value. Assortativity undefined.")
                results[attr].append(np.nan)
        else:
            print(f"Attribute '{attr}' not found or empty.")
            results[attr].append(np.nan)

    except Exception as e:
        print(f"Error processing {name}: {e}. Skipping network.")
        if len(results["network_size"]) > len(results.get("degree", [])):
             results["network_size"].pop()
        if name in network_names_processed:
             network_names_processed.remove(name)


# --- Plotting --- 
print("\nGenerating plots...")
network_sizes = np.array(results["network_size"])

all_attributes = list(attributes_to_analyze.keys()) + ["degree"]
plot_titles = list(attributes_to_analyze.values()) + ["Degree"]

for attr, title in zip(all_attributes, plot_titles):
    if attr not in results or not results[attr]:
        print(f"No results found for attribute '{attr}'. Skipping plots.")
        continue
        
    assortativity_values = np.array(results[attr])
    
    # Filter out NaN values for plotting
    valid_indices = ~np.isnan(assortativity_values)
    
    if not np.any(valid_indices):
        print(f"No valid (non-NaN) data to plot for {title}. Skipping plots.")
        continue
        
    valid_sizes = network_sizes[valid_indices]
    valid_values = assortativity_values[valid_indices]
    
    if len(valid_values) == 0:
        print(f"No valid data points after filtering NaNs for {title}. Skipping plots.")
        continue

    fig_scatter, ax_scatter = plt.subplots(figsize=(8, 6))
    ax_scatter.scatter(valid_sizes, valid_values, alpha=0.7)
    ax_scatter.axhline(0, color='red', linestyle='--', linewidth=0.8, label='No Assortativity (r=0)') 
    ax_scatter.set_title(f"Assortativity by {title} vs. Network Size (LCC)")
    ax_scatter.set_xlabel("Network Size (Number of Nodes in LCC)")
    ax_scatter.set_ylabel(f"Assortativity Coefficient (r) by {title}")
    ax_scatter.set_xscale('log')
    ax_scatter.grid(True, which="both", ls="--", linewidth=0.5)
    ax_scatter.legend()
    scatter_plot_path = os.path.join(output_dir, f"assortativity_vs_size_{attr}.png")
    try:
        fig_scatter.savefig(scatter_plot_path)
        print(f"Saved scatter plot for {title} to {scatter_plot_path}")
    except Exception as plot_err:
        print(f"Error saving scatter plot for {title}: {plot_err}")
    plt.close(fig_scatter)


    fig_hist, ax_hist = plt.subplots(figsize=(8, 6))
    ax_hist.hist(valid_values, bins=10, alpha=0.7, density=True)
    ax_hist.axvline(0, color='red', linestyle='--', linewidth=0.8, label='No Assortativity (r=0)') 
    ax_hist.set_title(f"Distribution of Assortativity Coefficients by {title}")
    ax_hist.set_xlabel(f"Assortativity Coefficient (r) by {title}")
    ax_hist.set_ylabel("Density")
    ax_hist.grid(True, axis='y', ls="--", linewidth=0.5)
    ax_hist.legend()
    hist_plot_path = os.path.join(output_dir, f"assortativity_distribution_{attr}.png")
    try:
        fig_hist.savefig(hist_plot_path)
        print(f"Saved histogram for {title} to {hist_plot_path}")
    except Exception as plot_err:
        print(f"Error saving histogram for {title}: {plot_err}")
    plt.close(fig_hist)


