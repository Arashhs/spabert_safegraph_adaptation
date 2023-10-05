from transformers.models.bert.modeling_bert import BertForMaskedLM
from transformers import BertTokenizer
import sys
sys.path.append("/panfs/jay/groups/28/yaoyi/kim01479/haystac/spabert/")

from models.spatial_bert_model import SpatialBertConfig, SpatialBertModel, SpatialBertForMaskedLM
from spabert.datasets.planetsense_loader import PlanetsenseDataset
from utils.common_utils import load_spatial_bert_pretrained_weights, get_spatialbert_embedding

import torch
import os
import tqdm

weight_path = '/panfs/jay/groups/28/yaoyi/kim01479/haystac/model_weights/singapore/mlm_mem_keeppos_ep9_iter20044_0.1064.pth'

config = SpatialBertConfig()
model = SpatialBertModel(config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

model = load_spatial_bert_pretrained_weights(model, weight_path)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

planetsense_dataset = PlanetsenseDataset(
    data_file_path = '/panfs/jay/groups/28/yaoyi/kim01479/haystac/spabert_planetsense_singapore.json',
    tokenizer = tokenizer,
    max_token_len = 512, 
    distance_norm_factor = 0.0001, 
    spatial_dist_fill=100
)

entity_ids = []
entity_names = []
entity_coords = []
entity_emb = []

for entity in planetsense_dataset:
    spabert_emb = get_spatialbert_embedding(entity, model)
    entity_ids.append(entity['pivot_id'])
    entity_names.append(entity['pivot_name'])
    entity_coords.append(entity['pivot_pos'])
    entity_emb.append(spabert_emb)
    
torch.save({"pivot_id": entity_ids, "pivot_name": entity_names, "pivot_pos":entity_coords, "spabert_emb": entity_emb}, '/panfs/jay/groups/28/yaoyi/kim01479/haystac/spabert_planetsense_singapore.pt')