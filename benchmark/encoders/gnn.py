import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import numpy as np
from dgl.nn import EGNNConv
from dgl.dataloading import GraphDataLoader


#### workflow ####
# directory of files from step 6 (pyg graphs)
# ...
# some other preprocessing steps
# ...
# ImmunoPred ... (includes preprocess_graphs and ??)
# then GraphDataLoader() to get data_loader for inference() 

DEVICE = 'cpu'

class GraphDataset(Dataset):
    def __init__(self, dataset):
        self.graph_data = dataset["graph_data"]
        self.labels = dataset["y"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        graph_data = self.graph_data[idx]
        y = self.labels[idx]
        return graph_data, y

def preprocess_graphs(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.pt')]

    # Initialize an empty list to store the graphs
    graphs = []

    # Loop through the files and load each graph, showing a progress bar
    for file in tqdm(files, desc="Loading graphs"):
        file_path = os.path.join(directory, file)
        graph = torch.load(file_path)
        graphs.append(graph)

    print(f"Loaded {len(graphs)} graphs.")

    graphs = [x for x in graphs if ('NXVPMVATV' not in x.name) and ('X' not in x.name)] # what does this do?

    all_graphs = []
    names = set()

    for graph in graphs:
        if graph.name not in names:
            names.add(graph.name)
            all_graphs.append(graph)

    #cut off h-bonding features for now
    for data in all_graphs:
        data.x = data.x[:, :-2]

    return all_graphs
# need to append immunogenicity score ?? ie target ??


def inference(model, data_loader, device):
    model.eval()
    true_targets = []
    predicted_probs = []  # Store raw probabilities for ROC AUC calculation

    with torch.no_grad():
        for graph_data in data_loader:
            graph_data = graph_data.to(device)   # correct ??

            final_output = model(graph_data)

            # Convert to probabilities
            # probs = torch.sigmoid(final_output).squeeze() # not necessary, my model has sigmoid after final layer
            probs = final_output.squeeze()

            # Handle the case where probs is a scalar
            if probs.ndim == 0:
                probs = probs.unsqueeze(0)  # Make it a 1-element tensor

            probs = probs.detach().cpu().numpy()

            true_targets.extend(target.detach().cpu().numpy())
            predicted_probs.extend(probs.tolist())  # Convert to list before extending

    return probs


class GraphModel(nn.Module):
    def __init__(self,
                 device,
                 gcn_layers: int = 5,
                 gat_hidden_channels: int = 64):
        super().__init__()

        self.device = device
        self.gat_hidden_channels = gat_hidden_channels

        self.GCN_layers = nn.ModuleList([EGNNConv(20, gat_hidden_channels, gat_hidden_channels, 1)])
        for _ in range(gcn_layers):
            self.GCN_layers.append(EGNNConv(gat_hidden_channels, gat_hidden_channels, gat_hidden_channels, 1))

        self.classif = nn.Sequential(
            # nn.Flatten(1),
            nn.Linear(gat_hidden_channels, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, graph_data):
        x, node_feat, coord_feat, edge_feat = graph_data, graph_data.ndata['x'][:, :20], graph_data.ndata['x'][:, 20:], graph_data.edata['edge_attr']

        for layer in self.GCN_layers:
            node_feat, coord_feat = layer(x, node_feat, coord_feat, edge_feat)

        # MLP classifier
        logits = self.classif(node_feat)

        return logits 


# what is the dimension of the output 
# do I need to flatten? no output is 1D vector of length 64
# what actually are the hidden channels?

# what is graph_data? 

def main():
    os.chdir('/home/josh/Dev/Project/encoders/ImmunoStruct')

    all_graphs = preprocess_graphs('./graph_pyg_IEDB')
    
    dataset = {"graph_data": all_graphs, "y": np.ones(len(all_graphs))}
    data_loader = GraphDataset(dataset)

    example = all_graphs[0]
    # print(f"{example.x}\n{example.edge_index}")
    #print(dataset["graph_data"][1], "\n", dataset["y"][1])
    print(dataset["graph_data"][1].x)
    print(dataset["graph_data"][1].coords) # coord_feat?
    print(dataset["graph_data"][1].node_id) # node_feat?
    print(dataset["graph_data"][1].edge_index) # edge_feat?
    # what are edge_feat, coord_feat, node_feat?


    # test_split_dataset can be 
    # test_loader = GraphDataLoader(test_split_dataset, batch_size=config.batch_size, collate_fn=collate, shuffle=False, num_workers=config.num_workers)

    # check what graph_data is model(graph_data)
    data_loader = GraphDataLoader(all_graphs)
    for graph_data in data_loader:
        x, node_feat, coord_feat, edge_feat = graph_data, graph_data.ndata['x'][:, :20], graph_data.ndata['x'][:, 20:], graph_data.edata['edge_attr']
        print(x)
        print(node_feat)
        print(coord_feat)
        print(edge_feat)

    #gnn = GraphModel(DEVICE)

    #inference(gnn, data_loader, DEVICE)

if __name__ == '__main__':
    main()



##############################################################################
################# PULL RELEVANT CODE FROM THIS MODEL #########################
##############################################################################

import torch
import torch.nn as nn

class HybridModelv2(nn.Module):
    def __init__(self,
                 vae_input_dim,
                 device,
                 gcn_layers: int = 5,
                 vae_hidden_dim: int = 512,
                 vae_latent_dim: int = 32,
                 gat_hidden_channels: int = 64,
                 self_attention_heads: int = 1,
                 property_embedding_dim: int = 8,
                 combined_attention_heads: int = 8,
                 *args, **kwargs):
        super().__init__()

        self.device = device
        self.vae_hidden_dim = vae_hidden_dim
        self.vae_latent_dim = vae_latent_dim
        self.vae_input_dim = vae_input_dim
        self.gat_hidden_channels = gat_hidden_channels
        self.property_embedding_dim = property_embedding_dim

        self.GCN_layers = nn.ModuleList([EGNNConv(20, gat_hidden_channels, gat_hidden_channels, 1)])
        for _ in range(gcn_layers):
            self.GCN_layers.append(EGNNConv(gat_hidden_channels, gat_hidden_channels, gat_hidden_channels, 1))

        # GAT components
        self.self_attention = MultiHeadAttention(gat_hidden_channels, self_attention_heads)

        # VAE components
        self.vae_fc1 = nn.Linear(vae_input_dim, vae_hidden_dim)
        self.vae_fc21 = nn.Linear(vae_hidden_dim, vae_latent_dim)  # Mean
        self.vae_fc22 = nn.Linear(vae_hidden_dim, vae_latent_dim)  # Log variance
        self.vae_fc3 = nn.Linear(vae_latent_dim + property_embedding_dim, vae_hidden_dim)
        self.vae_fc4 = nn.Linear(vae_hidden_dim, vae_input_dim)

        self.combined_attention = MultiHeadAttention(16, combined_attention_heads, input_dim=1)

        # Fusion/ Classifier layers
        self.classifier = self.get_classifier()

        self.property_embedding = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(32, self.property_embedding_dim),
            nn.ReLU(True)
        )

    def get_classifier(self):
        return nn.Sequential(
            nn.Flatten(1),
            nn.Linear(self.vae_latent_dim + self.property_embedding_dim + self.gat_hidden_channels, 32),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    def encode_vae(self, x):
        h1 = F.relu(self.vae_fc1(x))
        return self.vae_fc21(h1), self.vae_fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_vae(self, z):
        h3 = F.relu(self.vae_fc3(z))
        return self.vae_fc4(h3)  # Sigmoid for reconstruction

    def load_trained(self, path, new_head=False, map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location))
        if new_head:
            self.classifier = self.get_classifier().to(self.device)

    def forward(self, graph_data, sequence_data, peptide_property, return_embedding=False, return_attention=False):
        x, node_feat, coord_feat, edge_feat = graph_data, graph_data.ndata['x'][:, :20], graph_data.ndata['x'][:, 20:], graph_data.edata['edge_attr']

        # Create batch tensor based on the number of nodes in each graph
        # Assuming 'graph_data' is a batched DGL graph
        batch_tensor = torch.cat([torch.full((1, n), i) for i, n in enumerate(graph_data.batch_num_nodes())], dim=0)
        batch_tensor = batch_tensor.to(graph_data.device)

        for layer in self.GCN_layers:
            node_feat, coord_feat = layer(x, node_feat, coord_feat, edge_feat)

        node_feat = node_feat.view(batch_tensor.shape[0], -1, self.gat_hidden_channels)
        attention_output_n, attention_weights_n = self.self_attention(node_feat)
        attention_output_n = attention_output_n.view(-1, self.gat_hidden_channels)

        # Use global_mean_pool with the batch tensor
        x_gat_node = global_mean_pool(attention_output_n, batch_tensor.flatten())

        # peptide property
        peptide_property = self.property_embedding(peptide_property)

        # VAE part
        mu, logvar = self.encode_vae(sequence_data.view(-1, self.vae_input_dim))  # Flatten sequence input
        z_vae = self.reparameterize(mu, logvar)
        z_vae = torch.cat([z_vae, peptide_property], dim=1)
        recon_x = self.decode_vae(z_vae)


        # Fusion and final layers
        combined_x = torch.cat([x_gat_node, z_vae], dim=1)
        combined = torch.unsqueeze(combined_x, 2)
        combined, _ = self.combined_attention(combined)
        combined = torch.mean(combined, dim=2)

        combined_gat_only = torch.cat([x_gat_node], dim=1)

        final_output = self.classifier(combined)

        if return_embedding:
            return combined_gat_only, mu, logvar, final_output

        if return_attention:
            return attention_weights_n, mu, logvar, final_output

        return recon_x, mu, logvar, final_output