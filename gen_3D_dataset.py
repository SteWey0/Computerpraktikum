import numpy as np
rng = np.random.default_rng()
import os
import torch_geometric as tg
import torch
from tqdm import trange
from scipy.spatial import KDTree

class ThreeDGraphDataset(tg.data.Dataset):
    '''
    This class bundles the creation and saving as well as loading of a dataset of 3D graphs. If an instance is created, the class will 
    check in '3D_graphs' directory if the dataset is already processed. If not, the process() method will be called. Furthermore, the
    dataset will be loaded. If the dataset shall be calculated again, the process() method must be called explicitely.
    '''
    def __init__(self, root, n_graphs_per_type=100, transform=None, pre_transform=None):
        '''
        Args:
        - root (str): The directory where the dataset should be stored, divided into processed and raw dirs
        '''
        self.root = root
        self.n_graphs_per_type = n_graphs_per_type
        super().__init__(root, transform, pre_transform)
        
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
        Note: This does smh not work, therefore files are ATM recalculated every time.
        '''
        return ['data_00000.pt']
    
    def download(self):
        '''
        Download not implemented.
        '''
        pass
    
    def len(self):
        '''
        Returns the number of graphs in the dataset.
        '''
        return len([f for f in os.listdir(os.path.join(self.root, 'processed')) if f.startswith('data')])
    
    def get(self, idx):
        '''
        Returns the graph at index idx. 
        '''
        data = torch.load(os.path.join(self.processed_dir, 'data_{:05d}.pt'.format(idx)))
        return data
    
    def process(self):
        '''
        Here creation, processing and saving of the dataset happens. 
        '''
        # Some attributes for all graphs:
        self.size = np.array([10,10,10])
        lattice_types = {
             0: {'name': 'aP', 'nodes': self._get_P_nodes, 'binding_angles': [  0,   0,   0], 'scale': [0, 0, 0], 'label': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
             1: {'name': 'mP', 'nodes': self._get_P_nodes, 'binding_angles': [ 90,   0,  90], 'scale': [0, 0, 0], 'label': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
             2: {'name': 'mS', 'nodes': self._get_S_nodes, 'binding_angles': [ 90,   0,  90], 'scale': [0, 0, 0], 'label': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
             3: {'name': 'oP', 'nodes': self._get_P_nodes, 'binding_angles': [ 90,  90,  90], 'scale': [0, 0, 0], 'label': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
             4: {'name': 'oS', 'nodes': self._get_S_nodes, 'binding_angles': [ 90,  90,  90], 'scale': [0, 0, 0], 'label': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
             5: {'name': 'oI', 'nodes': self._get_I_nodes, 'binding_angles': [ 90,  90,  90], 'scale': [0, 0, 0], 'label': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]},
             6: {'name': 'oF', 'nodes': self._get_F_nodes, 'binding_angles': [ 90,  90,  90], 'scale': [0, 0, 0], 'label': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]},
             7: {'name': 'tP', 'nodes': self._get_P_nodes, 'binding_angles': [ 90,  90,  90], 'scale': [1, 1, 0], 'label': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]},
             8: {'name': 'tI', 'nodes': self._get_I_nodes, 'binding_angles': [ 90,  90,  90], 'scale': [1, 1, 0], 'label': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]},
             9: {'name': 'hR', 'nodes': self._get_P_nodes, 'binding_angles': [  0,   0,   0], 'scale': [1, 1, 1], 'label': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]},
            10: {'name': 'hP', 'nodes': self._get_P_nodes, 'binding_angles': [ 90,  90, 120], 'scale': [1, 1, 0], 'label': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]},
            11: {'name': 'cP', 'nodes': self._get_P_nodes, 'binding_angles': [ 90,  90,  90], 'scale': [1, 1, 1], 'label': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]},
            12: {'name': 'cI', 'nodes': self._get_I_nodes, 'binding_angles': [ 90,  90,  90], 'scale': [1, 1, 1], 'label': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]},
            13: {'name': 'cF', 'nodes': self._get_F_nodes, 'binding_angles': [ 90,  90,  90], 'scale': [1, 1, 1], 'label': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]},
        }

        
        for n in trange(self.n_graphs_per_type * 14):
            # Get graph features:
            pos, edge_index, label = self._process_lattice(lattice_types[n % 14])
            node_attr = self._get_node_attr(pos, edge_index)
            edge_attr = self._get_edge_attr(pos, edge_index)
            # Create data object:
            data = tg.data.Data(x          = torch.tensor(node_attr, dtype=torch.float), 
                                edge_index = torch.tensor(edge_index, dtype=torch.int64), 
                                edge_attr  = torch.tensor(edge_attr, dtype=torch.float), 
                                y          = torch.tensor(label, dtype=torch.float), 
                                pos        = torch.tensor(pos, dtype=torch.float))
            # Save data object:
            torch.save(data, os.path.join(self.processed_dir, 'data_{:05d}.pt'.format(n)))


    def _get_P_nodes(self, angles=np.array([90,90,90])):
        '''
        Get the nodes of a primitive lattice.
        '''
        scaling = np.sin(np.radians(angles))
        vec1 = np.arange(0,self.size[0],1)
        vec2 = np.arange(0,self.size[1]*scaling[2],1*scaling[2])
        vec3 = np.arange(0,self.size[2]*scaling[1],1*scaling[1])
        a, b, c = np.meshgrid(vec1,vec2, vec3)
        nodes = np.stack([a,b,c],axis=-1) # Stack them in a new axis
        nodes = np.reshape(nodes, (-1, 3)) # Reshape to an arr of nodes with shape (#nodes, 3)
        return nodes

    def _get_S_nodes(self):
        '''
        Get the nodes of a base-centred lattice.
        '''
        P = self._get_P_nodes()
        extra = P + np.array([0.5,0.5,0])
        return np.append(P,extra,axis=0)

    def _get_I_nodes(self):
        '''
        Get the nodes of a body-centred lattice.
        '''
        P = self._get_P_nodes()
        extra = P + 0.5
        return np.append(P,extra,axis=0)

    def _get_F_nodes(self):
        '''
        Get the nodes of a face-centred lattice.
        '''
        P = self._get_P_nodes()
        extra1 = P + np.array([0.5,0.5,0])
        extra2 = P + np.array([0,0.5,0.5])
        extra3 = P + np.array([0.5,0,0.5])
        return np.row_stack((P, extra1, extra2, extra3))
    

    def _process_lattice(self, arg_dict):
        '''
        Method that processes a lattice of a given type. The method is called with a dictionary holding parameters for one of the lattice types. It contains the following keys:
            - name: The name of the lattice type
            - nodes: The method to get the fitting fundamental lattice nodes
            - binding_angles: A list of binding angles [alpha, beta, gamma] of the lattice type. Angles are in degrees. 0° means to generate a independent random angle (0,180)°
            - scale: A list of scaling factors [x,y,z] for the lattice type. 0 means to generate a random scaling factor (0,2)
            - label: One hot encoded label for the lattice type
        '''
        # Get lattice angles
        angles = np.array(arg_dict['binding_angles'])
        if arg_dict['name'] == 'hR':
            # Special case for hR lattice as it has 3 identical but random angles
            angles = np.where(angles == 0, rng.uniform(46,89,1), angles)
        else:
            angles = np.where(angles == 0, rng.uniform(46,89,3), angles)
            
        # Get the fundamental lattice nodes
        if arg_dict['name'] in ['hR', 'hP']:
            # For hR and hP lattices we need to give the angles to the nodes method so that sheared connections are equally long
            nodes = arg_dict['nodes'](angles)
        else:
            nodes = arg_dict['nodes']()
        nodes = self._shear_nodes(nodes, angles)
        # Find random scale and apply gaussian noise to the lattice accordingly
        scale = np.array(arg_dict['scale'])
        scale = np.where(scale == 0, rng.uniform(0.1,3,3), scale)
        noise_level = 0.05 / scale  # At this step we scale the noise down, so that the scaling later on does not affect the noise level
        nodes += rng.normal(0, noise_level, nodes.shape)
        # Find the connections between the nodes in a given radius
        cons= self._get_cons_in_radius(nodes, 1.2+np.mean(noise_level))
        # Apply the saved scaling
        nodes *= scale
        
        # Add defects to the lattice
        #nodes, cons = self._add_defects(nodes, cons)
        return nodes, cons, np.array([arg_dict['label']])

    def _get_cons_in_radius(self, nodes):
        '''
        Get the connections in a radius as well as the total number of cons for each node.
        '''
        tree = KDTree(nodes)
        cons = tree.query_pairs(radius, output_type='ndarray', p=2)
        cons = cons.T
        cons = np.column_stack((cons, cons[::-1])) # Add the reverse connections
        return cons

    def _shear_nodes(self, nodes, binding_angle):
        '''
        Shear nodes. Binding angle is a 3D vector with the Binding angle in each axis.
        '''
        delta = np.tan(np.radians(np.array(binding_angle)))
        assert not np.any(delta == 0), 'Binding angle cannot be 0'
        nodes = nodes.astype(float)
        nodes = nodes + np.stack((nodes[:,2]/delta[0] + nodes[:,1]/delta[2], nodes[:,2]/delta[1] , np.zeros_like(nodes[:,1])), axis=1)
        return nodes

    def _add_defects(self, nodes, edge_index):
        '''
        Method that adds up to 10% of random defects (i.e. missing nodes) to the lattice. Should be called after _get_*_graph() but before
        _get_edge_attr() and _get_node_attr().
        '''
        # Draw up to 10% of unique random indices for nodes to be removed
        drop_indices = rng.choice(np.arange(len(nodes)), rng.integers(len(nodes)//10), replace=False)
        # Remove the nodes
        nodes = np.delete(nodes, drop_indices, axis=0)
        # Delete every connection that refers to a removed node
        edge_index = np.delete(edge_index, np.where(np.isin(edge_index, drop_indices))[1], axis=1)
        
        # As edge_index refers to the original node indices, we need to adjust the indices of most connections
        # For this we create a mapping from old indices to new indices
        old_to_new = np.arange(len(nodes) + len(drop_indices))  # Start with an array of original indices; [0,1,2,3,4,5,...]
        old_to_new[drop_indices] = -1  # Mark the indices of the nodes to be deleted; eg. drop_indices = [1,3] -> [0,-1,2,-1,4,5,...]
        old_to_new = np.cumsum(old_to_new != -1) - 1  # Create a cumulative sum array; cumsum([True, False, True, False, True, True,...]) -1 -> [1,1,2,2,3,4,...] -1 -> [0,0,1,1,2,3,...]
        
        # # Update edge indices to reflect new node indices through broadcasting magic
        edge_index = old_to_new[edge_index]
        return nodes, edge_index
        
    def _get_node_attr(self,nodes,cons):
        '''
        Method that returns the node attributes for each node in the graph. Should be called after creating the graph and adding defects.
        Returns an array of shape (len(pos) = #Nodes) with the entries [C] for each node.
            - C: Number of connections to other nodes
        '''
        # Get the number of connections for each node
        connection_counts = np.zeros(len(nodes))
        for edge in cons[0]:
            # Iterate over all edge start points and count the connections for each node. Start points sufficient, as connections are bidirectional.
            connection_counts[edge] += 1 
            
        return np.expand_dims(connection_counts, axis=1)
    
    def _get_edge_attr(self,nodes,cons):
        '''
        Method that returns the edge attributes for each edge in the graph. Should be called after creating the graph and adding defects.
        Returns an array of shape (len(edge_index[0])= #Edges, 2) with the entries [dx,dy] for each edge.
        '''
        # Get the edge vectors for each edge
        edge_vectors = nodes[cons[0]] - nodes[cons[1]]
        return edge_vectors
    