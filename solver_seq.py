# Some code based on https://github.com/thuml/Anomaly-Transformer

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from utils.utils import *
from model.Transformer import TransformerVar
from model.ModernTCN import Model
from model.loss_functions import *
from data_factory.data_loader import get_loader_segment
import logging
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from utils.MemTracker import MemTracker
import inspect
import math


os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'

def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class TwoEarlyStopping:
    def __init__(self, patience=10, verbose=False, dataset_name='', delta=0, type=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.val_loss2_min = np.inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2

class OneEarlyStopping:
    def __init__(self, patience=10, verbose=False, dataset_name='', delta=0, type=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.dataset = dataset_name
        self.type = type

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + f'_checkpoint_{self.type}.pth'))
        self.val_loss_min = val_loss

class Config:
    def __init__(self, items):
        for key, value in items.items():
            setattr(self, key, value)
class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):
        
        self.__dict__.update(Solver.DEFAULTS, **config)
        self.configs = Config(config)
        self.train_loader, self.vali_loader, self.k_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                               mode='train',
                                               dataset=self.dataset,backbone=self.backbone)

        self.test_loader, _ = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='test',
                                              dataset=self.dataset,backbone=self.backbone)
        self.thre_loader = self.vali_loader
        
        if self.memory_initial == "False":
            
            self.memory_initial = False
        else:
            self.memory_initial = True


        self.memory_init_embedding = None


        self.build_model(memory_init_embedding=self.memory_init_embedding)
        

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.entropy_loss = EntropyLoss()
        self.criterion = nn.MSELoss()

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
    

        formatter = logging.Formatter('%(asctime)s - %(message)s')
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)
        file_handler = logging.FileHandler(f'./number_{self.dataset}.log')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def build_model(self,memory_init_embedding):
        if "TCN" in self.backbone:
            self.model = Model( configs=self.configs, memory_initial=self.memory_initial, memory_init_embedding=memory_init_embedding, phase_type=self.phase_type, dataset_name=self.dataset)
        else:
            self.model = TransformerVar(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, configs=self.configs, \
                                    e_layers=3, d_model=self.d_model, n_memory=self.n_memory, device=self.device, backbone=self.backbone,\
                                    memory_initial=self.memory_initial, memory_init_embedding=memory_init_embedding, phase_type=self.phase_type, dataset_name=self.dataset)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            self.model = torch.nn.DataParallel(self.model, device_ids=[0], output_device=0).to(self.device)

    def vali(self, vali_loader):
        self.model.eval()

        valid_loss_list = [] ; valid_re_loss_list = [] ; valid_entropy_loss_list = []

        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            output_dict = self.model(input)
            output, queries, mem_items, attn = output_dict['out'], output_dict['queries'], output_dict['mem'], output_dict['attn']
            
            rec_loss = self.criterion(output, input)
            entropy_loss = self.entropy_loss(attn)
            loss = rec_loss + self.lambd * entropy_loss

            valid_re_loss_list.append(rec_loss.detach().cpu().numpy())
            valid_entropy_loss_list.append(entropy_loss.detach().cpu().numpy())
            valid_loss_list.append(loss.detach().cpu().numpy())

        return np.average(valid_loss_list), np.average(valid_re_loss_list), np.average(valid_entropy_loss_list)
    
    def train(self, training_type):

        print("======================TRAIN MODE======================")

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = OneEarlyStopping(patience=3, verbose=True, dataset_name=self.dataset, type=training_type)
        train_steps = len(self.train_loader)

        from tqdm import tqdm
        for epoch in tqdm(range(self.num_epochs)):
            iter_count = 0
            loss_list = []
            rec_loss_list = [] 
            entropy_loss_list = []

            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels) in enumerate(self.train_loader):
                
                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)    
                output_dict = self.model(input_data)
                
                output, memory_item_embedding, queries, mem_items, attn = output_dict['out'], output_dict['memory_item_embedding'], output_dict['queries'], output_dict["mem"], output_dict['attn']
                # print(memory_item_embedding.shape)
                rec_loss = self.criterion(output, input)
                entropy_loss = self.entropy_loss(attn)
                loss = rec_loss + self.lambd * entropy_loss
                
                loss_list.append(loss.detach().cpu().numpy())
                entropy_loss_list.append(entropy_loss.detach().cpu().numpy())
                rec_loss_list.append(rec_loss.detach().cpu().numpy())

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                try:
                    loss.mean().backward()
                    
                except:
                    import pdb; pdb.set_trace()
                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            train_loss = np.average(loss_list)
            train_entropy_loss = np.average(entropy_loss_list)
            train_rec_loss = np.average(rec_loss_list)

            valid_loss , valid_re_loss_list, valid_entropy_loss_list = self.vali(self.vali_loader)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, valid_loss))
            print(
                "Epoch: {0}, Steps: {1} | VALID reconstruction Loss: {3:.7f} Entropy loss Loss: {2:.7f}  ".format(
                    epoch + 1, train_steps, valid_re_loss_list, valid_entropy_loss_list))
            print(
                "Epoch: {0}, Steps: {1} | TRAIN reconstruction Loss: {3:.7f} Entropy loss Loss: {2:.7f}  ".format(
                    epoch + 1, train_steps, train_rec_loss, train_entropy_loss))
            print("tel:",train_entropy_loss,"vel:",valid_entropy_loss_list,"vl:",valid_loss,"trl:",train_loss)
            early_stopping(valid_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        return memory_item_embedding
    
    def test(self):
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint_second_train.pth')))
        self.model.eval()
        
        print("======================TEST MODE======================")

        gathering_loss = GatheringLoss(reduce=False)
        temperature = self.temperature

        train_attens_energy = []
        for i, (input_data, labels) in enumerate(self.train_loader):
            input = input_data.float().to(self.device)
            output_dict = self.model(input_data)
            output, queries, mem_items = output_dict['out'], output_dict['queries'], output_dict['mem']

            rec_loss = torch.mean(abs(input-output),dim=-1)
            latent_score = torch.softmax(gathering_loss(queries, mem_items)/temperature, dim=-1)
            # loss = latent_score * rec_loss
            loss = rec_loss

            cri = loss.detach().cpu().numpy()
            train_attens_energy.append(cri)

        train_attens_energy = np.concatenate(train_attens_energy, axis=0).reshape(-1)
        train_energy = np.array(train_attens_energy)

        valid_attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            output_dict = self.model(input)
            output, queries, mem_items = output_dict['out'], output_dict['queries'], output_dict['mem']

            rec_loss = torch.mean(abs(input-output),dim=-1)
            latent_score = torch.softmax(gathering_loss(queries, mem_items)/temperature, dim=-1)
            # loss = latent_score * rec_loss
            loss = rec_loss

            cri = loss.detach().cpu().numpy()
            valid_attens_energy.append(cri)

        valid_attens_energy = np.concatenate(valid_attens_energy, axis=0).reshape(-1)
        valid_energy = np.array(valid_attens_energy)

        combined_energy = np.concatenate([train_energy, valid_energy], axis=0)
        
        thresh = np.percentile(combined_energy, 100 - self.anomaly_ratio)
        print("Threshold :", thresh)
        # init_thresh = np.median(combined_energy)+10*(np.percentile(combined_energy, 75) - np.percentile(combined_energy, 25))
        init_thresh = np.mean(combined_energy)+3*np.std(combined_energy)

        distance_with_q = []
        reconstructed_output = []
        original_output = []
        rec_loss_list = []

        test_labels = []
        test_attens_energy = np.array([])
        score_threshs = []
        cache_threshs = []
        for i, (input_data, labels) in enumerate(self.test_loader):
            input = input_data.float().to(self.device)
            output_dict= self.model(input)
            output, queries, mem_items = output_dict['out'], output_dict['queries'], output_dict['mem']

            rec_loss = torch.mean(abs(input-output),dim=-1)
            # latent_score = torch.softmax(gathering_loss(queries, mem_items)/temperature, dim=-1)
            # loss = latent_score * rec_loss
            loss = rec_loss
            cri = loss.detach().cpu().numpy()
            test_attens_energy = np.append(test_attens_energy, cri[:,-1])###
            # print(test_attens_energy.shape)
            # score = np.array(test_attens_energy[-self.score_window:])
            # cache = np.array(test_attens_energy[-self.cache_window:])
            # score_thresh = np.median(score)+4*(np.percentile(score, 75) - np.percentile(score, 25))
            # cache_thresh = np.median(cache)+4*(np.percentile(cache, 75) - np.percentile(cache, 25))
            # score_threshs.append(score_thresh)
            # cache_threshs.append(cache_thresh)
            test_labels.append(labels[:,-1])

            # d_q = gathering_loss(queries, mem_items)*rec_loss
            # distance_with_q.append(d_q.detach().cpu().numpy())
            # distance_with_q.append(gathering_loss(queries, mem_items).detach().cpu().numpy())

            reconstructed_output.append(output[:,-1:,:].detach().cpu().numpy())
            original_output.append(input[:,-1:,:].detach().cpu().numpy())
            rec_loss_list.append(rec_loss.detach().cpu().numpy())
        # thresh = np.zeros(len(test_attens_energy))
        # for i in range(len(test_attens_energy)):
        #     if i < self.score_window:
        #         thresh[i] = init_thresh
        #         continue
        #     # score_thresh = np.median(test_attens_energy[i-self.score_window:i])+10*(np.percentile(test_attens_energy[i-self.score_window:i], 75) - np.percentile(test_attens_energy[i-self.score_window:i], 25))
        #     score_thresh = np.mean(test_attens_energy[i-self.score_window:i])+3*np.std(test_attens_energy[i-self.score_window:i])
        #     if i < self.cache_window:
        #         thresh[i] = score_thresh if score_thresh > init_thresh else init_thresh
        #     else:
        #         # cache_thresh = np.median(test_attens_energy[i-self.cache_window:i])+10*(np.percentile(test_attens_energy[i-self.cache_window:i], 75) - np.percentile(test_attens_energy[i-self.cache_window:i], 25))
        #         cache_thresh = np.mean(test_attens_energy[i-self.cache_window:i])+3*np.std(test_attens_energy[i-self.cache_window:i])
        #         thresh[i] = score_thresh if score_thresh > cache_thresh else cache_thresh
        # thresh = np.array(thresh)
        # test_attens_energy = np.concatenate(test_attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(test_attens_energy)
        test_labels = np.array(test_labels)

        reconstructed_output = np.concatenate(reconstructed_output,axis=0).reshape(-1)
        original_output = np.concatenate(original_output,axis=0).reshape(-1)
        rec_loss_list = np.concatenate(rec_loss_list,axis=0).reshape(-1)


        #reconstruct_path = f"./hyperparameters_tuning/reconstruction/{self.dataset}_"
        #np.save(reconstruct_path+'reconstructed_output', reconstructed_output)
        #np.save(reconstruct_path+'original_output', original_output)
        #np.save(reconstruct_path+'rec_loss',rec_loss_list)
        #np.save(reconstruct_path+'gt_labels',test_labels)
        #np.save(reconstruct_path+'anomaly_score_only_gathering_loss',test_energy)
        
        # distance_with_q = np.concatenate(distance_with_q,axis=0).reshape(-1)

        # normal_dist = []
        # abnormal_dist = []
        # for i,l in enumerate(test_labels):
        #     if l == 0:
        #         normal_dist.append(distance_with_q[i])
        #     else:
        #         abnormal_dist.append(distance_with_q[i])

        #dist_path = f"./hyperparameters_tuning/norm_abnorm_distribtuion/{self.dataset}_"
        #normal_dist = np.array(normal_dist)
        #abnormal_dist = np.array(abnormal_dist)

        #np.save(dist_path+'normal_dist_only_gl', normal_dist)
        #np.save(dist_path+'abnormal_dist_only_gl', abnormal_dist)
        print(test_energy.shape)
        pred = (test_energy > thresh).astype(int)

        gt = test_labels.astype(int)
        def visualize_energy(energy,pred,gt,orgin,recons,thresh,path):
            orgin = orgin.reshape(-1,self.input_c)
            recons = recons.reshape(-1,self.input_c)
            orgin = orgin[:,0].reshape(-1)
            recons = recons[:,0].reshape(-1)
            import matplotlib.pyplot as plt
            plt.clf()
            plt,axs = plt.subplots(nrows=4,ncols=1,figsize=(300,20))
            plt.subplots_adjust(left=0.1,right=0.9,top=0.9,bottom=0.1,hspace=0.4,wspace=0.4)
            axs[0].plot(range(len(orgin)),orgin,label='gt_value')
            gt = np.where(gt == 1)
            axs[0].scatter(gt[0],orgin[gt[0]],c='r')
            axs[0].legend()
            axs[1].plot(range(len(orgin)),orgin)
            pred = np.where(pred == 1)
            axs[1].scatter(pred[0],orgin[pred[0]],c='r')
            axs[2].plot(range(len(energy)),energy)
            try:
                axs[2].plot(range(len(thresh)),thresh)
            except:
                temp = np.zeros_like(energy)
                temp[:] = thresh
                axs[2].plot(range(len(energy)),temp)
            axs[3].plot(range(len(orgin)),orgin)
            axs[3].plot(range(len(recons)),recons)
            plt.tight_layout()
            plt.savefig(path)
            plt.clf()
            
        visualize_energy(test_energy,pred,gt,original_output,reconstructed_output,thresh,"./figs/test_energy_{0}.png".format(self.dataset))

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)
        if self.metrics == 'PA':
            anomaly_state = False
            for i in range(len(gt)):
                if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                    anomaly_state = True
                    for j in range(i, 0, -1):
                        if gt[j] == 0:
                            break
                        else:
                            if pred[j] == 0:
                                pred[j] = 1
                    for j in range(i, len(gt)):
                        if gt[j] == 0:
                            break
                        else: 
                            if pred[j] == 0:
                                pred[j] = 1
                elif gt[i] == 0:
                    anomaly_state = False
                if anomaly_state:
                    pred[i] = 1

            pred = np.array(pred)
            gt = np.array(gt)
            print("pred: ", pred.shape)
            print("gt:   ", gt.shape)
            

            accuracy = accuracy_score(gt, pred)
            precision, recall, f1_score, support = precision_recall_fscore_support(gt, pred,
                    
                                                                    average='binary')
        elif self.metrics == "AF":
            from utils.affiliation.generics import convert_vector_to_events
            from utils.affiliation.metrics import pr_from_events
            def getAffiliationMetrics(label, pred):
                events_pred = convert_vector_to_events(pred)
                events_label = convert_vector_to_events(label)
                Trange = (0, len(pred))

                result = pr_from_events(events_pred, events_label, Trange)
                P = result['precision']
                R = result['recall']
                F = 2 * P * R / (P + R)

                return P, R, F
            precision, recall, f1_score = getAffiliationMetrics(gt.copy(), pred.copy())
        print(" Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format( precision, recall, f1_score))
        print('='*50)

        self.logger.info(f"Dataset: {self.dataset}")
        self.logger.info(f"number of items: {self.n_memory}")
        self.logger.info(f"Precision: {round(precision,4)}")
        self.logger.info(f"Recall: {round(recall,4)}")
        self.logger.info(f"f1_score: {round(f1_score,4)} \n")
        # return  precision, recall, f1_score

    def get_memory_initial_embedding(self,training_type='second_train'):

        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint_first_train.pth')))
        self.model.eval()
        import sys
        k_output = []
        for i, (input_data, labels) in enumerate(self.k_loader):
            
            input = input_data.float().to(self.device)
            with torch.no_grad():
                if i==0:
                    output= self.model(input)['queries']
                    output = torch.cat([output[0],output[1:,-1,:]],dim=0)
                else:
                    out = self.model(input)['queries']
                    output = torch.cat((output,out[:,-1,:]),dim=0)
        self.memory_init_embedding = k_means_clustering(x=output, n_mem=self.n_memory, d_model=self.dims[-1]*(self.win_size//self.patch_stride))

        self.memory_initial = False

        self.build_model(memory_init_embedding = self.memory_init_embedding.detach())

        memory_item_embedding = self.train(training_type=training_type)

        memory_item_embedding = memory_item_embedding[:int(self.n_memory),:]

        item_folder_path = "memory_item"
        if not os.path.exists(item_folder_path):
            os.makedirs(item_folder_path)

        item_path = os.path.join(item_folder_path, str(self.dataset) + '_memory_item.pth')

        torch.save(memory_item_embedding, item_path)