import numpy as np
rng = np.random.default_rng()

def get_lattice_vectors(scale, binding_angle, initial_length=1):
    '''
    Function that returns the lattice vectors of a 2D lattice. It takes the following arguments:
    - scale: The scale of the lattice vectors (e.g. 2 means one lattice vector is twice as long as the initial_length)
    - binding_angle: The angle between two connections (e.g. 120 degrees for a hexagonal lattice)
    - initial_length: The length of the x lattice vector
    '''
    e1 = np.array([initial_length,0])
    e2 = np.array([initial_length*np.sin(np.radians(binding_angle-90)),initial_length*np.cos(np.radians(binding_angle-90))])
    e2 = e2*scale
    return e1, e2

def get_nodes(e1, e2, size=[10,10]):
    '''
    Function that returns the array of nodes in the 2D lattice. It takes the following arguments:
    - e1 (arr): The first lattice vector
    - e2 (arr): The second lattice vector
    - size (tuple): The size of the lattice in nodes
    '''
    nodes = []
    for i in range(size[0]):
        for j in range(size[1]):
            node = j*e1+i*e2
            nodes.append([node[0], node[1]])
    nodes = np.array(nodes)
    return nodes

def add_noise(nodes, noise_level=0.1):
    '''
    Function that adds noise to the nodes of a lattice. It takes the following arguments:
    - nodes (arr): The array of nodes
    - noise_level: The standard deviation of the noise
    '''
    noise = rng.normal(0, noise_level, nodes.shape)
    return nodes + noise

def get_connections(nodes, size):
    '''
    Function that returns the bidirectional connections in the lattice. Works for every lattice except hexagonal. 
    Arguments:
    - nodes (arr): The array of nodes
    - size (tuple): The size of the lattice in nodes
    Returns:
    - edges (arr): The array of connections in shape (2, n_edges)
    '''
    max_index = len(nodes)
    edges = []
    for i, node in enumerate(nodes):
        if i+1 < max_index and (i+1)%size[1] != 0:
            edges.extend([[i, i+1], [i+1, i]])
        if i+size[1] < max_index:
            edges.extend([[i, i+size[1]], [i+size[1], i]])
    return np.array(edges).T

def get_hex_connections(nodes, size):
    '''
    Function that returns the bidirectional connections in a hexagonal lattice. 
    Arguments:
    - nodes (arr[tuple]): The array of nodes
    - size (tuple): The size of the lattice in nodes
    Returns:
    - edges (arr): The array of connections in shape (2, n_edges)
    '''
    max_index = len(nodes)
    edges = []
    for i, node in enumerate(nodes):
        if (i+1)%size[1] != 0:
            # There is a node to the right
            edges.extend([[i, i+1], [i+1, i]])
        if i+size[1] < max_index:
            # There is a node above
            edges.extend([[i, i+size[1]], [i+size[1], i]])
        if i%size[1] != 0 and i+size[1]-1 < max_index:
            # There is a node to the upper left
            edges.extend([[i, i+size[1]-1], [i+size[1]-1, i]])
    return np.array(edges).T
    
def get_number_nn(nodes, connections):
    '''
    Function that returns the number of nearest neighbors for each node. It takes the following arguments:
    - nodes (arr): The array of nodes
    - connections (arr): The array of connections
    '''
    connections = connections.T
    nn = []
    for i in range(len(nodes)):
        nn.append(len([edge for edge in connections if i in edge]))
    return np._ArrayComplex_co(nn)/2

if __name__ == "__main__":
    # params
    size = [10,10]
    n_graphs = 100
    noise_level = 0.1
    
    # Generate square lattices
    scale = 1
    angle = 90
    e1, e2 = get_lattice_vectors(scale, angle, 1)
    nodes = np.array([get_nodes(e1,e2, size) for i in range(n_graphs)])
    nodes = add_noise(nodes, noise_level)
    connections = np.array([get_connections(nodes, size) for i in range(n_graphs)])
    nn = np.array([get_number_nn(nodes[i], connections[i]) for i in range(n_graphs)])
    np.savez(f'graphs/sq.npz', attr=nn, coords=nodes, edge_attr=[], edges=connections)
        
    # Generate rectangular lattices
    scale = 1
    angle = 90
    e1, e2 = get_lattice_vectors(scale, angle, 1)
    nodes = np.array([get_nodes(e1,e2, size) for i in range(n_graphs)])
    nodes = add_noise(nodes, noise_level)
    connections = np.array([get_connections(nodes, size) for i in range(n_graphs)])
    nn = np.array([get_number_nn(nodes[i], connections[i]) for i in range(n_graphs)])
    np.savez(f'graphs/rect.npz', attr=nn, coords=nodes, edge_attr=[], edges=connections)

    # Generate hexagonal lattices
    scale = 1
    angle = 120
    e1, e2 = get_lattice_vectors(scale, angle, 1)
    nodes = np.array([get_nodes(e1,e2, size) for i in range(n_graphs)])
    nodes = add_noise(nodes, noise_level)
    connections = np.array([get_hex_connections(nodes, size) for i in range(n_graphs)])
    nn = np.array([get_number_nn(nodes[i], connections[i]) for i in range(n_graphs)])
    np.savez(f'graphs/hex.npz', attr=nn, coords=nodes, edge_attr=[], edges=connections)
    
    print('Done!')
