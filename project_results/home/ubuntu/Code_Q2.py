import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter


output_dir = "/home/ubuntu/Results_Q2"
os.makedirs(output_dir, exist_ok=True)


files = os.listdir("/home/ubuntu/NET4103/fb100/data")


results = {}

for name in files:
    print(f"Processing {name} network...")
    try:
        G_full = nx.read_gml(f"/home/ubuntu/NET4103/fb100/data/{name}", label='id')
        print(f"{name} (Full): Nodes={G_full.number_of_nodes()}, Edges={G_full.number_of_edges()}")

        connected_components = sorted(nx.connected_components(G_full), key=len, reverse=True)
        if not connected_components:
             print(f"Error: No connected components found for {name}.")
             continue
        lcc_nodes = connected_components[0]
        G = G_full.subgraph(lcc_nodes).copy()
        print(f"{name} (LCC): Nodes={G.number_of_nodes()}, Edges={G.number_of_edges()}")

        results[name] = {'graph': G}

        # --- Q2(a): Degree Distribution --- 
        degrees = [d for n, d in G.degree()]
        degree_counts = Counter(degrees)
        deg, cnt = zip(*sorted(degree_counts.items()))
        
        results[name]['degrees'] = degrees
        results[name]['degree_distribution'] = (deg, cnt)

        fig_deg, ax_deg = plt.subplots(figsize=(8, 6))
        total_nodes = G.number_of_nodes()
        prob = [c / total_nodes for c in cnt]
        ax_deg.plot(deg, prob, 'o-')
        ax_deg.set_title(f"Degree Distribution for {name} (LCC)")
        ax_deg.set_xlabel("Degree (k)")
        ax_deg.set_ylabel("Probability P(k)")
        ax_deg.set_xscale('log')
        ax_deg.set_yscale('log')
        ax_deg.grid(True, which="both", ls="--", linewidth=0.5)
        fig_deg_path = os.path.join(output_dir, f"{name.replace(' ', '_')}_degree_distribution.png")
        fig_deg.savefig(fig_deg_path)
        plt.close(fig_deg)
        print(f"Saved degree distribution plot to {fig_deg_path}")
        results[name]['degree_plot'] = fig_deg_path

        # --- Q2(b): Clustering Coefficients and Density --- 
        global_clustering = nx.transitivity(G)
        mean_local_clustering = nx.average_clustering(G) 
        edge_density = nx.density(G)
        
        results[name]['global_clustering'] = global_clustering
        results[name]['mean_local_clustering'] = mean_local_clustering
        results[name]['edge_density'] = edge_density
        print(f"{name} Metrics:")
        print(f"Global Clustering (Transitivity): {global_clustering:.4f}")
        print(f"Mean Local Clustering: {mean_local_clustering:.4f}")
        print(f"Edge Density: {edge_density:.6f}")

        # --- Q2(c): Degree vs. Local Clustering Coefficient --- 
        local_clustering = nx.clustering(G)
        node_degrees = dict(G.degree())
        
        degrees_list = [node_degrees[node] for node in G.nodes()]
        local_clustering_list = [local_clustering[node] for node in G.nodes()]
        
        results[name]['local_clustering'] = local_clustering
        results[name]['degree_vs_clustering_data'] = (degrees_list, local_clustering_list)

        fig_clust, ax_clust = plt.subplots(figsize=(8, 6))
        ax_clust.scatter(degrees_list, local_clustering_list, alpha=0.5, s=10) 
        ax_clust.set_title(f"Degree vs. Local Clustering for {name} (LCC)")
        ax_clust.set_xlabel("Degree (k)")
        ax_clust.set_ylabel("Local Clustering Coefficient")
        ax_clust.set_xscale('log') 
        ax_clust.grid(True, which="both", ls="--", linewidth=0.5)
        fig_clust_path = os.path.join(output_dir, f"{name.replace(' ', '_')}_degree_vs_clustering.png")
        fig_clust.savefig(fig_clust_path)
        plt.close(fig_clust)
        print(f"Saved degree vs. clustering plot to {fig_clust_path}")
        results[name]['degree_vs_clustering_plot'] = fig_clust_path

    except Exception as e:
        print(f"Error processing {name}: {e}")
