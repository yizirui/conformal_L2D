import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassifierRejector(nn.Module):
    def __init__(self, base_model, num_classes, n_features, with_softmax=True, decouple=False):
        super(ClassifierRejector, self).__init__()
        self.base_model_clf = base_model
        base_mdl_lst = [self.base_model_clf]
        if decouple:
            self.base_model_rej = copy.deepcopy(self.base_model_clf)
            base_mdl_lst += [self.base_model_rej]
        else:
            self.base_model_rej = self.base_model_clf

        self.fc_clf = nn.Linear(n_features, num_classes)
        self.fc_clf.bias.data.zero_()

        self.fc_rej = nn.Linear(n_features, 1)
        self.fc_rej.bias.data.zero_()

        self.with_softmax = with_softmax
        self.params = nn.ModuleDict({
            'base': nn.ModuleList(base_mdl_lst),
            'clf' : nn.ModuleList([self.fc_clf,self.fc_rej])
        })

    def forward(self, x):
        out = self.base_model_clf(x)
        logits_clf = self.fc_clf(out) # [B,K]

        out = self.base_model_rej(x)
        logit_rej = self.fc_rej(out) # [B,1]

        out = torch.cat([logits_clf,logit_rej], -1) # [B,K+1]

        if self.with_softmax:
            out = F.softmax(out, dim=-1)
        return out
