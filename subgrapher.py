import numpy as np
import networkx as nx
import random

### Graph generation functions ###
def generate_k_regular_digraph(n, k):
    """
    Generates a directed graph with `n` nodes and a fixed in- and out-degree of `k` for each node.

    Parameters:
    -----------
    n : int
        The number of nodes in the graph.
    k : int
        The out-degree of each node in the graph.

    Returns:
    --------
    networkx.DiGraph
        A directed graph with `n` nodes and a fixed in- and out-degree of `k` for each node.
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    for node in G.nodes():
        potential_targets = list(G.nodes())
        potential_targets.remove(node)  # Remove the current node to avoid self-loops

        # Remove nodes that already have an edge from current node
        potential_targets = [t for t in potential_targets if not G.has_edge(node, t)]

        # Randomly select k distinct targets
        targets = random.sample(potential_targets, k)
        for target in targets:
            G.add_edge(node, target)

    # Reassign edges for nodes that exceed in-degree of k
    for node in G.nodes():
        while len(G.in_edges(node)) > k:
            # Randomly select an incoming edge to reassign
            source_to_reassign, _ = random.choice(list(G.in_edges(node)))
            G.remove_edge(source_to_reassign, node)

            # Find a new target for the edge
            potential_new_targets = list(G.nodes())
            potential_new_targets.remove(source_to_reassign)
            potential_new_targets = [t for t in potential_new_targets if not G.has_edge(source_to_reassign, t) and len(G.in_edges(t)) < k]

            if not potential_new_targets:
                raise ValueError("Cannot generate the desired graph with given n and k.")

            new_target = random.choice(potential_new_targets)
            G.add_edge(source_to_reassign, new_target)

    return G


def add_edges_with_preferential_attachment(G, additional_edges_per_node):
    """
    Adds edges to the graph `G` using the preferential attachment algorithm. For each node in `G`, `additional_edges_per_node`
    edges are added to the graph, with the probability of connecting to a node proportional to its degree.

    Parameters:
    -----------
    G : networkx.Graph
        The graph to which to add edges.
    additional_edges_per_node : int
        The number of edges to add to each node in `G`.

    Returns:
    --------
    networkx.DiGraph
        The graph `G` with `additional_edges_per_node` edges added to each node by PA.
    """
    for node in G.nodes():
        for _ in range(additional_edges_per_node):
            # Choose target based on in-degree
            target = random.choices(
                population=list(G.nodes()),
                weights=[G.in_degree(n) + 1 for n in G.nodes()],  # +1 to ensure non-zero probability
                k=1
            )[0]

            while target == node or G.has_edge(node, target):
                target = random.choices(
                    population=list(G.nodes()),
                    weights=[G.in_degree(n) + 1 for n in G.nodes()],
                    k=1
                )[0]

            G.add_edge(node, target)

    return G


def regular_digraph_plus_pa(n, min_indegree, fixed_outdegree):
    """
    Generates a directed graph with `n` nodes, a minimum in-degree of `min_indegree`, and a fixed out-degree of `fixed_outdegree`
    for each node. First a regular digraph is generated, and then edges are added according to preferential attachment.

    Parameters:
    -----------
    n : int
        The number of nodes in the graph.
    min_indegree : int
        The minimum in-degree of each node in the graph.
    fixed_outdegree : int
        The fixed out-degree of each node in the graph.

    Returns:
    --------
    networkx.DiGraph
        A directed graph with `n` nodes, a minimum in-degree of `min_indegree`, and a fixed out-degree of `fixed_outdegree`
        for each node.
    """
    G = generate_k_regular_digraph(n, min_indegree)
    G = add_edges_with_preferential_attachment(G, fixed_outdegree - min_indegree)
    return G


def random_network_fixed_outdegree(n, outdegree, min_indegree):
    """
    Generates a directed graph with `n` nodes, a fixed out-degree of `outdegree` for each node, and a minimum in-degree of
    `min_indegree` for each node. Edges are added according to preferential attachment, and then edges are rewired to
    ensure that the minimum in-degree is met.

    Parameters:
    -----------
    n : int
        The number of nodes in the graph.
    outdegree : int
        The fixed out-degree of each node in the graph.
    min_indegree : int
        The minimum in-degree of each node in the graph.

    Returns:
    --------
    networkx.DiGraph
        A directed graph with `n` nodes, a fixed out-degree of `outdegree` for each node, and a minimum in-degree of
        `min_indegree` for each node.
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    # Preferential attachment: probability proportional to in-degree
    for i in range(n):  
        nodes = sorted(set(G.nodes()) - {i})
        weights = [G.in_degree(node) + 1 for node in nodes]  # +1 to ensure non-zero probability
        weights = np.array(weights) / sum(weights)
        
        neighbors = np.random.choice(nodes, outdegree, replace=False, p=weights)
        
        for neighbor in neighbors:
            G.add_edge(i, neighbor)

    # Edge rewiring to ensure minimum in-degree
    low_indegree_nodes = [node for node, deg in G.in_degree() if deg < min_indegree]
    while low_indegree_nodes:
        for node in low_indegree_nodes:
            while G.in_degree(node) < min_indegree:
                # Find a node with in-degree > min_indegree and out-degree = outdegree
                suitable_nodes = [n for n, deg in G.in_degree() if deg > min_indegree and not G.has_edge(n, node)]
                if not suitable_nodes:
                    break
                source = np.random.choice(suitable_nodes)
                
                # Find an edge from the source node that can be rewired without creating a duplicate edge
                potential_targets = [target for target in G.successors(source) if not G.has_edge(node, target) and target != node]
                if not potential_targets:
                    continue
                target = np.random.choice(potential_targets)
                
                # Rewire the edge
                G.remove_edge(source, target)
                G.add_edge(source, node)
                ods = dict(G.out_degree()).values()
        low_indegree_nodes = [node for node, deg in G.in_degree() if deg < min_indegree]

    return G

### Helper functions ###

def safediv(x, y):
    """
    Computes the division of `x` by `y`, handling the case where `x` is zero or `y` is zero.

    Parameters:
    -----------
    x : float or int
        The numerator of the division.
    y : float or int
        The denominator of the division.

    Returns:
    --------
    float or int
        The result of the division, or 0 if `x` is zero, or `np.inf` if `y` is zero.
    """
    if x==0:
        return 0
    try:
        return x/y
    except ZeroDivisionError:
        return np.inf

def edge_scores(G, g_indeg, g_outdeg, h_indeg, h_outdeg, desired_k):
    """
    Computes a score for each edge in the graph `G` based on the in-degree and out-degree of the nodes in `G`, as well as
    the in-degree and out-degree of a target graph `H`. The weight of an edge is proportional to the product of the
    difference between the target degree and the actual degree of the nodes incident to the edge, divided by the product of
    the difference between the degree of the nodes in `G` and the target degree. The score is then normalised by the sum
    of all scores in `G`. The score is computed as follows:

    Score = (need for edge in H) / (availability of edge from G)
    Where: need for edge in H = (desired degree - in-degree of target node) * (desired degree - out-degree of source node)
    And: availability of edge from G = (in-degree of target node - desired degree) * (out-degree of source node - desired degree)

    Some safe division rules are also applied to handle infs and 0s. In these cases, the score is set to 1 or 0, as adding that edge is necessary or impossible.

    The score is used to determine which edge(s) to add to `H` next.

    Parameters:
    -----------
    G : networkx.DiGraph
        The graph for which to compute the edge weights.
    g_indeg : dict
        A dictionary mapping nodes in `G` to their in-degree.
    g_outdeg : dict
        A dictionary mapping nodes in `G` to their out-degree.
    h_indeg : dict
        A dictionary mapping nodes in the target graph `H` to their in-degree.
    h_outdeg : dict
        A dictionary mapping nodes in the target graph `H` to their out-degree.
    desired_k : int
        The desired in/out degree for the nodes in `H`.

    Returns:
    --------
    gw : dict
        A dictionary mapping edges in `G` to their weights.
    infs : bool
        A boolean indicating whether any of the edge weights are infinite.
    """
    # compute edge scores
    es = {e: (safediv((max(desired_k - h_indeg.get(e[1], 0), 0) *
              max(desired_k - h_outdeg.get(e[0], 0), 0)), 
            (max(g_indeg[e[1]] - desired_k, 0) *
                max(g_outdeg[e[0]] - desired_k, 0))))
        for e in G.edges()}
    infs = False
    # normalise
    tot_weight = sum(es.values())

    # account for infs (cases where the edge is necessary)
    if tot_weight == np.inf:
        es = {e: 1 if v==np.inf else 0 for e, v in es.items()}
        infs = True
    tot_weight = sum(es.values())

    # actually normalise
    es = {e: w/tot_weight for e, w in es.items()}
    return es, infs

def edge_swap(edges, G_toremove, G_toadd, d_in_toremove, d_out_toremove, d_in_toadd, d_out_toadd):
    """
    Removes the edges in `edges` from the graph `G_toremove`, and adds them to the graph `G_toadd`. Updates the in-degree and
    out-degree dictionaries `d_in_toremove`, `d_out_toremove`, `d_in_toadd`, and `d_out_toadd` accordingly.

    Parameters:
    -----------
    edges : list of tuples
        A list of tuples representing the edges to remove from `G_toremove` and add to `G_toadd`.
    G_toremove : networkx.DiGraph
        The graph from which to remove the edges.
    G_toadd : networkx.DiGraph
        The graph to which to add the edges.
    d_in_toremove : dict
        A dictionary mapping nodes in `G_toremove` to their in-degree.
    d_out_toremove : dict
        A dictionary mapping nodes in `G_toremove` to their out-degree.
    d_in_toadd : dict
        A dictionary mapping nodes in `G_toadd` to their in-degree.
    d_out_toadd : dict
        A dictionary mapping nodes in `G_toadd` to their out-degree.

    Returns:
    --------
    None
    """
    G_toremove.remove_edges_from(edges)
    G_toadd.add_edges_from(edges)
    for i, j in edges:
        d_in_toremove[j] -= 1
        d_out_toremove[i] -= 1
        d_in_toadd[j] += 1
        d_out_toadd[i] += 1       

def rand_edge_removal(G, H, last_edge, h_indeg, h_outdeg, g_indeg, g_outdeg, shuff):
    """
    Removes edges from the graph `H` based on their proximity to problematic `last_edge`.
    Chance of an edge being randomly removed is proportional to the path length from the source and target of `last_edge` to the edge.
    Updates the graphs `G` and `H`, as well as the degree dictionaries.
    `h_indeg`, `h_outdeg`, `g_indeg`, and `g_outdeg`.

    Parameters:
    -----------
    G : networkx.DiGraph
        The graph `G` with edges that could be added to `H`.
    H : networkx.DiGraph
        The `H` subgraph.
    last_edge : tuple
        A tuple representing the last edge that was added to `H`.
    h_indeg : dict
        A dictionary mapping nodes in `H` to their in-degree.
    h_outdeg : dict
        A dictionary mapping nodes in `H` to their out-degree.
    g_indeg : dict
        A dictionary mapping nodes in `G` to their in-degree.
    g_outdeg : dict
        A dictionary mapping nodes in `G` to their out-degree.
    shuff : float
        A float between 0 and 1 indicating the fraction of edges to shuffle before removing an edge.

    Returns:
    --------
    None
    """
    # get path length in H of every node to last_edge source and target
    source_paths = nx.single_source_shortest_path_length(H.reverse(copy=True),
                                                         last_edge[0])
    target_paths = nx.single_source_shortest_path_length(H, last_edge[1])

    # get weights for each edge in H
    w = {(s,t): 
         1/((source_paths.get(s, np.inf) + 1)*(target_paths.get(t, np.inf)+1)) ###?
        for s, t in H.edges()}
    
    #normalise
    total_weight = sum(w.values())
    if total_weight:
        w = {e: w[e]/total_weight for e in w}
    else:
        w = {e: 1/len(w) for e in w}

    # do not consider edges in disconnected components (they do not affect the error at hand)
    wnonzero = {e: w[e] for e in w if w[e] > 0}
    cands = list(w.keys())

    # Select edges to remove by score
    edges = np.random.choice(list(range(len(cands))),
                                min(int(shuff*len(H.edges())), len(wnonzero)),
                                replace=False, p=list(w.values()))
    edges = [cands[i] for i in edges]

    # Remove edges from H and restore to G   
    edge_swap(edges, H, G, h_indeg, h_outdeg, g_indeg, g_outdeg)


def shuffle(best_H, best_HG, last_edge, best_graph_degs, shuff):
    """
    Removes a fraction of edges from the graph `best_H`, and updates the graphs `H` and `G`, as well as the degree dictionaries
    `h_indeg`, `h_outdeg`, `g_indeg`, and `g_outdeg`.

    Parameters:
    -----------
    best_H : networkx.DiGraph
        The current best subgraph `H`.
    best_HG : networkx.DiGraph
        The corresponding graph `G` with edges from `H` removed.
    last_edge : tuple
        The last edge to be added to `H`, and potential source of the error.
    best_graph_degs : tuple
        A tuple of dictionaries containing the in-degree and out-degree of nodes in `best_H` and `best_HG`.
    shuff : float
        A float between 0 and 1 indicating the fraction of edges to shuffle before removing an edge.

    Returns:
    --------
    H : networkx.DiGraph
        The updated subgraph of `H`.
    G : networkx.DiGraph
        The updated subgraph of `G`.
    h_indeg : dict
        A dictionary mapping nodes in `H` to their in-degree.
    h_outdeg : dict
        A dictionary mapping nodes in `H` to their out-degree.
    g_indeg : dict
        A dictionary mapping nodes in `G` to their in-degree.
    g_outdeg : dict
        A dictionary mapping nodes in `G` to their out-degree.
    """
    # copy the best subgraph and corresponding graph
    H = best_H.copy()
    G = best_HG.copy()

    # copy the degree dictionaries
    h_indeg, h_outdeg, g_indeg, g_outdeg = best_graph_degs[0].copy(), best_graph_degs[1].copy(), best_graph_degs[2].copy(), best_graph_degs[3].copy()

    # remove a fraction of edges from H and restore to G
    rand_edge_removal(G, H, last_edge, h_indeg, h_outdeg,
                       g_indeg, g_outdeg, shuff)
    return H, G, h_indeg, h_outdeg, g_indeg, g_outdeg


def extract_regular_subgraph(G, desired_k, shuff0=0.01):
    """
    Attempts to extract a regular subgraph from the directed graph `G` with a desired in- and out-degree of `desired_k` for each node.

    Functions by iteratively adding edges to a subgraph `H` from `G` based on their calculated edge scores, until the desired degree is reached.
    If adding an edge would make the desired degree impossible, the edge is not added and a fraction of the subgraph edges removed and the process is repeated.

    Parameters:
    -----------
    G : networkx.DiGraph
        The directed graph from which to extract the subgraph.
    desired_k : int
        The desired out-degree of each node in the subgraph.
    shuff0 : float, optional
        The initial fraction of edges to shuffle before removing an edge. Default is 0.01.

    Returns:
    --------
    networkx.DiGraph
        A directed subgraph of `G` with a desired in- and out-degree of `desired_k` for each node.
    """
    
    # Initialise
    G0 = G.copy()
    target_edges = desired_k*len(G.nodes())
    H = nx.DiGraph()
    best_H = H.copy()
    g_indeg = dict(G.in_degree())
    g_outdeg = dict(G.out_degree())
    h_indeg = {x:0 for x in G.nodes()}
    h_outdeg = {x:0 for x in G.nodes()}
    ran_record = [[]]
    ess_record = []
    shuff = shuff0
    nn = 0
    nw = 0

    # check possible:
    if not (all([g_indeg[n] + h_indeg[n] >= desired_k for n in G.nodes()] +
            [g_outdeg[n] + h_outdeg[n] >= desired_k for n in G.nodes()])):
        raise ValueError("Impossible on this network")

    while len(H.edges()) < target_edges:
        nw+=1
        # randomly remove edges to not get stuck in local minima
        if nw%(target_edges//10) == 0:
            edges = random.sample(list(H.edges()), k=int(0.05*len(H.edges())))
            edge_swap(edges, H, G, h_indeg, h_outdeg, g_indeg, g_outdeg)

        # Get edge scores
        try:
            g_weights, infs = edge_scores(G, g_indeg, g_outdeg,
                                          h_indeg, h_outdeg, desired_k)
        except ZeroDivisionError:
            nn += 1
            print('Shuffling %d at %.2f%%. %d/%d edges assigned'
                %(nn, 100*shuff, len(best_H.edges()), desired_k*len(G.nodes())))
            last_edge = random.choice([x for x in ran_record if x][-1])
            H, G, h_indeg, h_outdeg, g_indeg, g_outdeg = shuffle(best_H, best_HG, last_edge,
                                                                 best_graph_degs, shuff)
            shuff *= 1.01 # increase amount of shuffling next step
            if shuff > 1: # if shuffling takes too long, give up
                raise ValueError("Timeout")
            ran_record.append([])
            continue
        
        # Determine which edge(s) to add next
        if infs: # Check if any necessary edges to add
            ran_record.append([])
            edges = [edge for edge, v in g_weights.items() if v]
            ess_record.append(edges)
        elif nn%2: # If not, alternate between weighted random and deterministic* max
            edges = random.choices(list(g_weights.keys()), k=1,
                                   weights=list(g_weights.values()))
            ran_record[-1].extend(edges)
        else: # deterministic max (with random choice if tied)
            maxw = max(g_weights.values())
            edges = [e for e, w in g_weights.items() if w == maxw]
            edges = random.choices(edges, k=1)
            ran_record[-1].extend(edges)
        
        # Add edges
        edge_swap(edges, G, H, g_indeg, g_outdeg, h_indeg, h_outdeg)

        # Check if adding necessary edges has made further solution impossible
        if any([x > desired_k for x in h_indeg.values()] +
            [x > desired_k for x in h_outdeg.values()]):
            print('Too many necessary edges')
            try:
                last_edge = random.choice([x for x in ran_record if x][-1])
            except IndexError:
                raise ValueError("Impossible(?) on this network")   
            nn += 1

            # Restore edges
            edge_swap(edges, H, G, h_indeg, h_outdeg, g_indeg, g_outdeg)

            # shuffle to try again
            print('Shuffling %d at %.2f%%. %d/%d edges assigned'
                %(nn, 100*shuff, len(best_H.edges()), desired_k*len(G.nodes())))
            H, G, h_indeg, h_outdeg, g_indeg, g_outdeg = shuffle(H, G, last_edge,
                                                                 best_graph_degs, shuff)
            ran_record.append([])
            shuff *= 1.01
            if shuff > 1:
                raise ValueError("Timeout")
        
        # if improvement, save
        if ((len(H.edges) >= len(best_H.edges)) &
            bool(set(H.edges()) - set(best_H.edges()))):
            shuff = shuff0 # reset shuffler for new graph solution
            best_H = H.copy()
            best_HG = G.copy()
            best_graph_degs = (h_indeg.copy(), h_outdeg.copy(),
                               g_indeg.copy(), g_outdeg.copy())

    print('Solution found')

    # verify that H is a subgraph of G0
    subgraph = set(H.edges()).issubset(set(G0.edges()))

    # verify no degree variation
    fh_indeg = dict(best_H.in_degree())
    fh_outdeg = dict(best_H.out_degree())
    min_indeg = min(fh_indeg.values())
    min_outdeg = min(fh_outdeg.values())
    max_indeg = max(fh_indeg.values())
    max_outdeg = max(fh_outdeg.values())
    indegc = min_indeg == max_indeg
    outdegc = min_outdeg == max_outdeg

    print('Subgraph = %s' %subgraph)
    print('In degree min = %d, max = %d'%(min_indeg, max_indeg))
    print('Out degree min = %d, max = %d'%(min_outdeg, max_outdeg))

    if any([not subgraph, not indegc, not outdegc]):
        raise ValueError("Solution not valid, fatal error")

    return best_H