import torch
from torch import nn
import pickle
import os
from Utils.TimeLogger import log


class ProteinTrainer:
    def __init__(self, data_handler, args):
        self.handler = data_handler
        self.args = args

        self.metrics = dict()
        mets = ['Loss', 'SeqAccuracy', 'PropertyPrediction']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()

    def run(self):
        self.prepare_model()
        log('Model Prepared')

        if self.args.load_model != None:
            self.load_model()

        for ep in range(self.args.epoch):
            train_results = self.train_epoch()
            log(self.make_print('Train', ep, train_results, True))

            if ep % self.args.test_epoch == 0:
                test_results = self.test_epoch()
                log(self.make_print('Test', ep, test_results, True))
                self.save_history()

        final_results = self.test_epoch()
        log(self.make_print('Test', self.args.epoch, final_results, True))
        self.save_history()

    def prepare_model(self):
        # 从Model.py的prepareModel方法改造
        self.model = ProteinGNNTransformer(
            num_proteins=self.handler.num_proteins,
            num_amino_acids=self.handler.num_amino_acids,
            num_properties=self.handler.num_properties,
            latent_dim=self.args.latent_dim
        ).cuda()

        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.criterion = nn.CrossEntropyLoss()

    # 其他函数如train_epoch, test_epoch, save_history, load_model等可以从Coach类中改造