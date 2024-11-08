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
        pass        
    def _get_square_graph(self):
        pass
    def _get_rect_graph(self):
        pass
    def _get_edge_attr(self, pos, edge_index):
        pass
    def _get_node_attr(self, pos, edge_index):
        pass
    