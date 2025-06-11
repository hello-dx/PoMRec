# -*- coding: UTF-8 -*-
# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

""" PoMRec
    python3 main.py --model_name PoMRec --dataset ml-1m --lr 0.001 --K 2 --prompt_num --lamb 1
"""

import os
import logging
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from models.BaseModel import SequentialModel
from utils import layers


class PoMRec(SequentialModel):
    reader = 'SeqReader'
    runner = 'BaseRunner'
    extra_log_args = []
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--attn_size', type=int, default=8,
                            help='Size of attention vectors.')
        parser.add_argument('--K', type=int, default=3,
                            help='Number of hidden interests.')
        parser.add_argument('--prompt_num', type=int, default=4,
                            help='Temperature in knowledge distillation loss.')
        parser.add_argument('--n_layers', type=int, default=1,
                            help='Number of the projection layer.')
        parser.add_argument('--lamb', type=float, default=3,
                            help='Training stage: pretrain / finetune.')
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.attn_size = args.attn_size
        self.K = args.K
        self.prompt_num = args.prompt_num
        self.n_layers = args.n_layers
        self.lamb = args.lamb
        self.max_his = args.history_max
        self._define_params()
        self.apply(self.init_weights)
        logging.info('Train from scratch!')

    def _define_params(self):
        self.interest_extractor = MultiInterestExtractor(self.K, self.item_num, self.emb_size,
             self.attn_size,self.max_his, self.prompt_num, self.lamb)
        self.proj = nn.Sequential()
        for i, _ in enumerate(range(self.n_layers - 1)):
            self.proj.add_module('proj_' + str(i), nn.Linear(self.emb_size, self.emb_size))
            self.proj.add_module('dropout_' + str(i), nn.Dropout(p=0.5))
            self.proj.add_module('relu_' + str(i), nn.ReLU(inplace=True))
        self.proj.add_module('proj_final', nn.Linear(self.emb_size, self.K))

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        model_dict = self.state_dict()
        state_dict = torch.load(model_path)
        exist_state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(exist_state_dict)
        self.load_state_dict(model_dict)
        logging.info('Load model from ' + model_path)

    @staticmethod
    def similarity(a, b):  # cosine similarity
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        return (a * b).sum(dim=-1)

    def forward(self, feed_dict):
        self.check_list = []
        i_ids = feed_dict['item_id']  # bsz, -1
        history = feed_dict['history_items']  # bsz, max_his
        lengths = feed_dict['lengths']  # bsz
        batch_size, seq_len = history.shape

        out_dict = dict()
        interest_vectors, distri_vectors = self.interest_extractor(history, lengths)
        i_vectors = self.interest_extractor.i_embeddings(i_ids)
        pred_intent = self.proj(distri_vectors)  # bsz, K
        # self.check_list.append(('pred_intent', pred_intent.softmax(-1)))

        user_vector = (interest_vectors * pred_intent.softmax(-1)[:, :, None]).sum(-2)  # bsz, emb
        prediction = (user_vector[:, None, :] * i_vectors).sum(-1)
        out_dict['prediction'] = prediction.view(batch_size, -1)

        return out_dict

    def loss(self, out_dict: dict):
        loss = super().loss(out_dict)
        return loss


class MultiInterestExtractor(nn.Module):
    def __init__(self, k, item_num, emb_size, attn_size, max_his, prompt_num, lamb):
        super(MultiInterestExtractor, self).__init__()
        self.K = k
        self.max_his = max_his
        self.prompt_num = prompt_num
        self.lamb = lamb
        self.emb_size = emb_size

        self.i_embeddings = nn.Embedding(item_num, emb_size)
        self.p_embeddings = nn.Embedding(max_his + 1, emb_size)

        self.max_prompt = 5
        self.prompt_pad = torch.ones(self.max_prompt-self.prompt_num, emb_size).cuda()
        self.prompt1 = nn.Embedding(self.prompt_num, emb_size)
        self.prompt2 = nn.Embedding(self.prompt_num, emb_size)

        self.W1 = nn.Linear(emb_size, attn_size)
        self.W2 = nn.Linear(attn_size, k)

        self.W3 = nn.Linear(emb_size, attn_size)
        self.W4 = nn.Linear(attn_size, 1)

        self.map = nn.Parameter(nn.init.xavier_normal_(
            torch.tensor(np.random.randn(2, 1), dtype=torch.float32, requires_grad=True)))

    def value2attn(self, values, mask):
        values = values.masked_fill(mask.unsqueeze(-1) == 0, -np.inf)
        values = values.transpose(-1, -2)
        attn = (values - values.max()).softmax(dim=-1)
        attn = attn.masked_fill(torch.isnan(attn), 0) # bsz, K, his_max
        return attn

    def forward(self, history, lengths):
        batch_size, seq_len = history.shape
        valid_his = (history > 0).long()
        self.history = history

        his_vectors = self.i_embeddings(history)
        len_range = torch.from_numpy(np.arange(self.max_his)).to(history.device)
        position = (lengths[:, None] - len_range[None, :seq_len]) * valid_his
        pos_vectors = self.p_embeddings(position)
        his_vectors = his_vectors + pos_vectors   # bsz, his_max, dim

        valid_his = torch.cat([valid_his, torch.ones([valid_his.shape[0], self.max_prompt]).cuda()], 1)
        # Multi-Interest Extraction
        prompt1 = torch.cat([self.prompt_pad, self.prompt1.weight])
        prompt1 = torch.tile(prompt1[None, :, :], [his_vectors.shape[0], 1, 1])
        his_vectors_prompt1 = torch.cat([his_vectors, prompt1], 1)
        attn_score = self.W2(self.W1(his_vectors_prompt1).tanh())  # bsz, his_max, K
        attn_score = self.value2attn(attn_score, valid_his)  # bsz, K, his_max
        interest_vectors = (his_vectors_prompt1[:, None, :, :] * attn_score[:, :, :, None]).sum(-2)  # bsz, K, emb
        self.mean = interest_vectors
        self.attn = attn_score
        var = []
        for k in range(self.K):
            x_mean_2 = (his_vectors_prompt1 - interest_vectors[:, k:k + 1, :]) ** 2  ## bsz, his_max+1, dim
            var_k = torch.matmul(attn_score[:, k:k + 1, :], x_mean_2)  ## bsz, 1, dim+1
            var_k = torch.sqrt(var_k)
            var.append(var_k)
        variance = torch.cat(var, 1)  ## bsz, k, dim
        self.std = variance
        interest_vectors = interest_vectors + self.lamb * variance

        # Interest Distribution Predict
        prompt2 = torch.cat([self.prompt_pad, self.prompt2.weight])
        prompt2 = torch.tile(prompt2[None, :, :], [his_vectors.shape[0], 1, 1])
        his_vectors_prompt2 = torch.cat([his_vectors, prompt2], 1)
        distri_pred = self.W4(self.W3(his_vectors_prompt2).tanh()) # bsz, his_max, 1
        distri_pred = self.value2attn(distri_pred, valid_his) # bsz, 1, his_max
        distri_vectors = torch.matmul(distri_pred, his_vectors_prompt2).squeeze()

        return interest_vectors, distri_vectors
