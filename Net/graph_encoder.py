import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
import json
import torch

class GNN(nn.Module):
    def __init__(self, edge_index, type_num, rel_num, emb_dim, emb_mid, emb_out, encoder='gcn', num_layers=2, num_heads=2, data_set='GDS'):  # added data_set
        super(GNN, self).__init__()
        self.type_num = type_num
        self.edge_index = edge_index
        self.node_embedding = nn.Embedding(type_num + rel_num, emb_dim)

        # use node embeddings obtained from LLM
        data_path = str(data_set)
        rel_path = f'./data/{data_path}/relation_vectors_with_categories.json'
        entype_path = f'./data/{data_path}/entype_vectors_with_categories.json'
        rel_id = f'./data/{data_path}/rel2id.json'
        entype_id = f'./data/{data_path}/type2id.json'
        with open(rel_path, 'r', encoding='utf-8') as f:
            rel_data = json.load(f)
        with open(entype_path, 'r', encoding='utf-8') as f:
            entype_data = json.load(f)
        with open(rel_id, 'r', encoding='utf-8') as f:
            rel_id_data = json.load(f)
        with open(entype_id, 'r', encoding='utf-8') as f:
            entype_id_data = json.load(f)
        rel_item = {}
        for item in rel_data:
            rel_vec = item['vector']
            rel_id = rel_id_data[item['category']]
            rel_data_temp = {str(rel_id): rel_vec}
            rel_item.update(rel_data_temp)
        entype_item = {}
        for item in entype_data:
            entype_vec = item['vector']
            entype_id = entype_id_data[item['category']]
            entype_data_temp = {str(entype_id): entype_vec}
            entype_item.update(entype_data_temp)
        sorted_rel = dict(sorted(rel_item.items(), key=lambda x: int(x[0])))
        val_rel = list(sorted_rel.values())
        rel_tensor = torch.tensor(val_rel)
        sorted_entype = dict(sorted(entype_item.items(), key=lambda x: int(x[0])))
        val_entype = list(sorted_entype.values())
        entype_tensor = torch.tensor(val_entype)
        rel_entype = torch.cat((rel_tensor, entype_tensor), dim=0)
        self.node_embedding = nn.Embedding.from_pretrained(rel_entype)

        if encoder.lower() == 'gcn':
            self.encoder = nn.ModuleList(
                [
                    GCNConv(emb_dim if i == 0 else emb_mid, emb_mid if i < num_layers - 1 else emb_out)
                    for i in range(num_layers)
                ]
            )
        elif encoder.lower() == "sage":
            self.encoder = nn.ModuleList(
                [
                    SAGEConv(
                        in_channels=emb_dim if i == 0 else emb_mid,
                        out_channels=emb_out if i == num_layers - 1 else emb_mid,
                    )
                    for i in range(num_layers)
                ]
            )

        elif encoder.lower() == "gat":
            assert emb_mid % num_heads == 0
            assert emb_out % num_heads == 0

            self.encoder = nn.ModuleList(
                [
                    GATConv(
                        in_channels=emb_dim if i == 0 else emb_mid,
                        out_channels=(emb_out // num_heads) if i == num_layers - 1 else (emb_mid // num_heads),
                        heads=num_heads,
                        fill_value="mean"
                    )
                    for i in range(num_layers)
                ]
            )
        self.act = nn.GELU()
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.node_embedding.weight)

    def forward(self):
        X = self.node_embedding.weight
        for i, layer in enumerate(self.encoder):
            X = self.act(layer(X, self.edge_index))
        Type, Rel = X[:self.type_num], X[self.type_num:]
        return Type, Rel

