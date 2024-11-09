import numpy as np
rng = np.random.default_rng()
import os
import torch_geometric as tg
import torch
from tqdm import trange


class PlaneGraphDataset(tg.data.Dataset):
    def __init__(self, root, n_graphs_per_type=100, transform=None, pre_transform=None):
        '''
        Args:
        - root (str): The directory where the dataset should be stored, divided into processed and raw dirs
        '''
        super().__init__(root, transform, pre_transform)
        self.root = root
        self.n_graphs_per_type = n_graphs_per_type
        
    @property
    def raw_file_names(self):
        '''
        If this file exists in the raw directory, the download will be skipped. Download not implemented.
        '''
        return 'raw.txt'
    
    @property
    def processed_file_names(self):
        '''
        If this file exists in the processed directory, processing will be skipped. 
        '''
        return ['data_0000.pt']
    
    def download(self):
        '''
        Download not implemented.
        '''
        pass
    
    def len(self):
        '''
        Returns the number of graphs in the dataset.
        '''
        return len(os.listdir(os.path.join(self.root, 'processed')))
    
    def get(self, idx):
        '''
        Returns the graph at index idx. 
        '''
        data = torch.load(os.path.join(self.processed_dir, 'data_{:04d}.pt'.format(idx)))
        return data
    
    def process(self):
        '''
        Here creation, processing and saving of the dataset happens. 
        '''
        # Some attributes for all graphs:
        self.size = [10,10]
        # Set standard graphs:
        self._set_standard_graphs()
        for n in trange(self.n_graphs_per_type * 3):
            # Get graph features:
            if n%3 == 0:
                # square lattice
                pos, edge_index = self._get_square_graph()
                edge_attr = self._get_edge_attr(pos, edge_index)
                node_attr = self._get_node_attr(pos, edge_index)
                label = torch.tensor([1,0,0])
            elif n%3 == 1:
                # rectangular lattice
                pos, edge_index = self._get_rect_graph()
                edge_attr = self._get_edge_attr(pos, edge_index)
                node_attr = self._get_node_attr(pos, edge_index)
                label = torch.tensor([0,1,0])
            else:
                # hexagonal lattice
                pos, edge_index = self._get_hex_graph()
                edge_attr = self._get_edge_attr(pos, edge_index)
                node_attr = self._get_node_attr(pos, edge_index)
                label = torch.tensor([0,0,1])
            
            # Create data object:
            data = tg.data.Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr, y=label, pos=pos)
            # Save data object:
            torch.save(data, os.path.join(self.processed_dir, 'data_{:04d}.pt'.format(n)))
    
    #TODO:
    # Idea is now differnet. Start by creating normal square and hex lattices with connections in ugly and inefficiert way.
    # Save as class attributes. In the _get_* methods these standard graphs are muted by simple and efficient array operations. See test.ipynb
    def _set_standard_graphs(self):
        '''
        This helper method is used to set standard square and hexagonal lattices with connections. 
        As this is inefficent, these are set once and later only modified by array operations.
        '''
        # Square
        nodes = []
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                node = j*np.array([1,0]) + i*np.array([0,1])
                nodes.append([node[0], node[1]])
        self.square_nodes = np.array(nodes)
        # Hexagonal
        e1 = np.array([1,0])
        e2 = np.array([1*0.5,1*(np.sqrt(3)/2)]) # cos(120°) and sin(120°) resp.
        nodes = []
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                node = j*e1 + i*e2
                nodes.append([node[0], node[1]])
        self.hex_nodes = np.array(nodes)
        
        # Connections for square lattice
        max_index = len(self.square_nodes)
        edges = []
        for i in range(max_index):
            if i+1 < max_index and (i+1)%self.size[1] != 0:
                edges.extend([[i, i+1], [i+1, i]])
            if i+self.size[1] < max_index:
                edges.extend([[i, i+self.size[1]], [i+self.size[1], i]])
        self.square_cons = np.array(edges).T
        # Connections for hexagonal lattice
        max_index = len(self.hex_nodes)
        edges = []
        for i in range(max_index):
            if (i+1)%self.size[1] != 0:
                # There is a node to the right
                edges.extend([[i, i+1], [i+1, i]])
            if i+self.size[1] < max_index:
                # There is a node above
                edges.extend([[i, i+self.size[1]], [i+self.size[1], i]])
            if i%self.size[1] != 0 and i+self.size[1]-1 < max_index:
                # There is a node to the upper left
                edges.extend([[i, i+self.size[1]-1], [i+self.size[1]-1, i]])
        self.hex_cons = np.array(edges).T      
          
    def _get_square_graph(self):
        '''
        Method that returns the position and connections of a square lattice. Applies randomly different sorts of noise to the "perfect" lattice.
        '''
        # Apply a random scaling of the lattice (but ensure squareness)
        scale = rng.uniform(0.5, 2)
        nodes = self.square_nodes*scale
        # Apply gaussian noise with random standard deviation
        noise_level = rng.uniform(0, 0.2)
        nodes += rng.normal(0, noise_level, nodes.shape)
        # Apply a random systematic skew, maybe later
        # skew_angle = rng.uniform(-10, 10)
        # nodes += np.stack((nodes[:,1]/np.tan(np.radians(90 + skew_angle)), np.zeros_like(nodes[:,1])), axis=1)
        
        # Set average distance between nodes for later use:
        self.avg_dist = scale+noise_level
        # For now, no alterations to connections
        connections = self.square_cons
        return nodes, connections
        
    def _get_rect_graph(self):
        # Apply a random scaling of the lattice that makes it rectangular
        scale = rng.uniform(0.5, 2, 2)
        nodes = self.square_nodes*scale
        # Apply gaussian noise with random standard deviation
        noise_level = rng.uniform(0, 0.2)
        nodes += rng.normal(0, noise_level, nodes.shape)
        # Apply a random systematic skew, maybe later
        # skew_angle = rng.uniform(-10, 10)
        # nodes += np.stack((nodes[:,1]/np.tan(np.radians(90 + skew_angle)), np.zeros_like(nodes[:,1])), axis=1)
        
        # Set average distance between nodes for later use:
        self.avg_dist = np.mean(scale)+noise_level
        # For now, no alterations to connections
        connections = self.square_cons
        return nodes, connections
    
    def _get_hex_graph(self):
        # Apply a random scaling of the lattice
        scale = rng.uniform(0.5, 2)
        nodes = self.hex_nodes*scale
        # Apply gaussian noise with random standard deviation
        noise_level = rng.uniform(0, 0.2)
        nodes += rng.normal(0, noise_level, nodes.shape)
        # Apply a random systematic skew, maybe later
        # skew_angle = rng.uniform(-10, 10)
        # nodes += np.stack((nodes[:,1]/np.tan(np.radians(90 + skew_angle)), np.zeros_like(nodes[:,1])), axis=1)
        
        # Set average distance between nodes for later use:
        self.avg_dist = scale+noise_level
        # For now, no alterations to connections
        connections = self.hex_cons
        return nodes, connections
    
    def _get_edge_attr(self, pos, edge_index):
        pass
    
    def _get_node_attr(self, pos, edge_index):
        # Get number of nodes within typical radius around each node
        diff = pos[:,np.newaxis,:] - pos[np.newaxis,:,:] # Use of broadcasting to get [4,4,2] array -> difference vectors between all 4x4 node pairs
        dist = np.linalg.norm(diff, axis=2) # Get the lenght of the difference vectors
        neighbor_counts = np.sum((dist <= self.avg_dist) & (dist > 0), axis=1) # Collapse the dist matrix and count the times where the distance is within the typical radius
        
        # Get the number of connections for each node
        connection_counts = np.zeros(len(pos))
        for edge in edge_index[0]:
            # Iterate over all edge start points and count the connections for each node. Start points sufficient, as connections are bidirectional.
            connection_counts[edge] += 1 
        return np.stack((neighbor_counts, connection_counts), axis=1)
    
    def _add_defects(self, pos, edge_index):
        # Draw up to 10% of unique random indices for nodes to be removed
        drop_indices = rng.choice(np.arange(len(nodes)), rng.integers(len(nodes)//10), replace=False)
        # Remove the nodes
        pos = np.delete(pos, drop_indices, axis=0)
        # Delete every connection that refers to a removed node
        edge_index = np.delete(edge_index, np.where(np.isin(edge_index, drop_indices))[1], axis=1)
        
        # As edge_index refers to the original node indices, we need to adjust the indices of most connections
        # For this we create a mapping from old indices to new indices
        old_to_new = np.arange(len(pos) + len(drop_indices))  # Start with an array of original indices; [0,1,2,3,4,5,...]
        old_to_new[drop_indices] = -1  # Mark the indices of the nodes to be deleted; eg. drop_indices = [1,3] -> [0,-1,2,-1,4,5,...]
        old_to_new = np.cumsum(old_to_new != -1) - 1  # Create a cumulative sum array; cumsum([True, False, True, False, True, True,...]) -1 -> [1,1,2,2,3,4,...] -1 -> [0,0,1,1,2,3,...]
        
        # # Update edge indices to reflect new node indices through broadcasting magic
        edge_index = old_to_new[edge_index]
        return pos, edge_index