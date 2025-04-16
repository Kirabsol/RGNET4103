import networkx as nx
from packaging import version
import sys 
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

Gcaltech = nx.read_gml("fb100/data/Caltech36.gml")
Gmit = nx.read_gml("fb100/data/MIT8.gml")
Gjohnshopkins = nx.read_gml("fb100/data/Johns Hopkins55.gml")

def plot_degree_distribution(G):

    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    hist, bin_edges = np.histogram(degree_sequence, density=True)

    plt.semilogy(bin_edges[:-1], hist, 'o', ms = 15)
    plt.xlabel(r"Degree")
    plt.ylabel(r"PDF")
    plt.title("Degree Distribution"+G.name())
    plt.show()

plot_degree_distribution(Gcaltech)
plot_degree_distribution(Gmit)
plot_degree_distribution(Gjohnshopkins)
