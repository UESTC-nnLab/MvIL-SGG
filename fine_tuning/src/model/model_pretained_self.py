from src.model.cos_sim import cosine_similarity_matrix
from src.model.model import visualize_similarity_matrix
from statictis_Triplet.stati_rel import get_valid_totalStati

if __name__ == '__main__' and __package__ is None:
    from os import sys

    sys.path.append('../')
import copy
import os
import wandb
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.dataset.DataLoader import (CustomDataLoader, collate_fn_mmg)
from src.dataset.dataset_builder import build_dataset
from src.model.SGFN_MMG.model import Mmgnet
from src.utils import op_utils
from src.utils.eva_utils_acc import get_mean_recall, get_zero_shot_recall
from src.model.SGFN_MMG.model1 import Mmgnet as Mmgnet1
import torch.nn.functional as F


class MMGNet():
    def __init__(self, config):
        self.config = config
        self.model_name = self.config.NAME
        self.mconfig = mconfig = config.MODEL
        self.exp = config.exp
        self.save_res = config.EVAL
        self.update_2d = config.update_2d

        ''' Build dataset '''
        dataset = None
        if config.MODE == 'train':
            if config.VERBOSE: print('build train dataset')
            self.dataset_train = build_dataset(self.config, split_type='train_scans', shuffle_objs=True,
                                               multi_rel_outputs=mconfig.multi_rel_outputs,
                                               use_rgb=mconfig.USE_RGB,
                                               use_normal=mconfig.USE_NORMAL)
            self.dataset_train.__getitem__(0)

        if config.MODE == 'train' or config.MODE == 'trace':
            if config.VERBOSE: print('build valid dataset')
            self.dataset_valid = build_dataset(self.config, split_type='validation_scans', shuffle_objs=False,
                                               multi_rel_outputs=mconfig.multi_rel_outputs,
                                               use_rgb=mconfig.USE_RGB,
                                               use_normal=mconfig.USE_NORMAL)
            dataset = self.dataset_valid

        if config.MODE == 'eval':
            if config.VERBOSE: print('build valid dataset')
            self.dataset_valid = build_dataset(self.config, split_type='validation_scans', shuffle_objs=False,
                                               multi_rel_outputs=mconfig.multi_rel_outputs,
                                               use_rgb=mconfig.USE_RGB,
                                               use_normal=mconfig.USE_NORMAL)
            dataset = self.dataset_valid

        num_obj_class = len(self.dataset_valid.classNames)
        num_rel_class = len(self.dataset_valid.relationNames)
        self.num_obj_class = num_obj_class
        self.num_rel_class = num_rel_class

        if config.MODE == 'train':
            self.total = self.config.total = len(self.dataset_train) // self.config.Batch_Size
            self.max_iteration = self.config.max_iteration = int(
                float(self.config.MAX_EPOCHES) * len(self.dataset_train) // self.config.Batch_Size)
            self.max_iteration_scheduler = self.config.max_iteration_scheduler = int(
                float(100) * len(self.dataset_train) // self.config.Batch_Size)

        if config.MODE == "eval":
            self.total = self.config.total = 0
            self.max_iteration = self.config.max_iteration = 0
            self.max_iteration_scheduler = self.config.max_iteration_scheduler = 0

        ''' Build Model '''
        self.model = Mmgnet(self.config, num_obj_class, num_rel_class,adapter=True).to(config.DEVICE)
        self.samples_path = os.path.join(config.PATH, self.model_name, self.exp, 'samples')
        self.results_path = os.path.join(config.PATH, self.model_name, self.exp, 'results')
        self.trace_path = os.path.join(config.PATH, self.model_name, self.exp, 'traced')
        self.writter = None

        if not self.config.EVAL:
            pth_log = os.path.join(config.PATH, "logs", self.model_name, self.exp)
            self.writter = SummaryWriter(pth_log)

    def load(self, best=False):
        return self.model.load(best)

    @torch.no_grad()
    def data_processing_train(self, items):
        obj_points, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids,_ = items
        obj_points = obj_points.permute(0, 2, 1).contiguous()
        obj_points, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids = \
            self.cuda(obj_points, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids)
        return obj_points, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids

    @torch.no_grad()
    def data_processing_train1(self, items):
        obj_points, rel_points, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids,_ = items
        obj_points = obj_points.permute(0, 2, 1).contiguous()
        rel_points = rel_points.permute(0, 2, 1).contiguous()
        obj_points,rel_points, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids = \
            self.cuda(obj_points,rel_points, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids)
        return obj_points,rel_points, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids
    @torch.no_grad()
    def data_processing_val(self, items):
        obj_points,rel_points, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids, scan_ids = items
        obj_points = obj_points.permute(0, 2, 1).contiguous()
        obj_points, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids = \
            self.cuda(obj_points, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids)
        return obj_points, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids, scan_ids

    def report_loss(self, all_datas, wandb_run):
        training_dict = {}

        for t_data in all_datas:
            training_dict[t_data[0]]=t_data[1]

        wandb_run.log(training_dict)

    def train(self):
        ''' create data loader '''
        drop_last = True
        train_loader = CustomDataLoader(
            config=self.config,
            dataset=self.dataset_train,
            batch_size=self.config.Batch_Size,
            num_workers=4,
            drop_last=drop_last,
            shuffle=True,
            collate_fn=collate_fn_mmg,
        )


        os.environ["WANDB_API_KEY"]='######################3'
        os.environ["WANDB_MODE"]="offline"

        run_id = wandb.util.generate_id()

        # wandb_run = wandb.init(project="SGG_VL",
        #                        entity="honsen",
        #                        group="ablation",
        #                        name="pretrain",
        #                        # resume="auto",
        #                        # id="3m5ba36f"
        #                        )

        if self.model.iteration > 1:
            self.model.epoch = self.model.iteration // self.total
        else:
            self.model.epoch = 1
        keep_training = True

        if self.total == 1:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return

        progbar = op_utils.Progbar(self.total, width=20, stateful_metrics=['Misc/epo', 'Misc/it', 'Misc/lr'])

        ''' Resume data loader to the last read location '''
        loader = iter(train_loader)
        if True:
            obj_encoder_dict = torch.load("obj_encoder.pth")
            self.model.obj_encoder.load_state_dict(obj_encoder_dict)

            rel_encoder_dict = torch.load("rel_encoder.pth")
            self.model.rel_encoder_3d.load_state_dict(rel_encoder_dict,strict=False)

            mmg_dict = torch.load("mmg.pth")
            self.model.mmg.load_state_dict(mmg_dict,strict=False)

            mlp_dict = torch.load("mlp.pth")
            self.model.mlp_3d.load_state_dict(mlp_dict)

        for k, p in self.model.named_parameters():
            if p.requires_grad:
                print(f"Para {k} need grad")
        ''' Train '''
        while (keep_training):

            if self.model.epoch > self.config.MAX_EPOCHES:
                break

            print('\n\nTraining epoch: %d' % self.model.epoch)

            for items in loader:
                self.model.train()

                ''' get data '''
                obj_points,rel_points, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids = self.data_processing_train1(
                    items)
                logs = self.model.process_train(obj_points, gt_class, descriptor, gt_rel_cls,
                                                edge_indices, batch_ids, with_log=True,
                                                weights_obj=self.dataset_train.w_cls_obj,
                                                weights_rel=self.dataset_train.w_cls_rel,
                                                ignore_none_rel=False)

                iteration = self.model.iteration
                logs += [
                    ("Misc/epo", int(self.model.epoch)),
                    ("Misc/it", int(iteration)),
                    ("lr", self.model.lr_scheduler.get_last_lr()[0])
                ]

                qwe = [x for x in logs if not x[0].startswith('Loss')]

                # self.report_loss(qwe, wandb_run)

                progbar.add(1, values=logs \
                    if self.config.VERBOSE else [x for x in logs if not x[0].startswith('Loss')])
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.log(logs, iteration)
                if self.model.iteration >= self.max_iteration:
                    break

            progbar = op_utils.Progbar(self.total, width=20, stateful_metrics=['Misc/epo', 'Misc/it'])
            loader = iter(train_loader)
            self.save()

            if ('VALID_INTERVAL' in self.config and self.config.VALID_INTERVAL > 0 and self.model.epoch % self.config.VALID_INTERVAL == 0):
                print('start validation...')
                rel_acc_val = self.validation()
                self.model.eva_res = rel_acc_val
                self.save()

            self.model.epoch += 1

    def cuda(self, *args):
        return [item.to(self.config.DEVICE) for item in args]

    def log(self, logs, iteration):
        # Tensorboard
        if self.writter is not None and not self.config.EVAL:
            for i in logs:
                if not i[0].startswith('Misc'):
                    self.writter.add_scalar(i[0], i[1], iteration)

    def save(self):
        self.model.save()


    def validation(self, debug_mode=False):
        val_loader = CustomDataLoader(
            config=self.config,
            dataset=self.dataset_valid,
            batch_size=1,
            num_workers=self.config.WORKERS,
            drop_last=False,
            shuffle=False,
            collate_fn=collate_fn_mmg
        )

        total = len(self.dataset_valid)
        progbar = op_utils.Progbar(total, width=20, stateful_metrics=['Misc/it'])
        
        print('===   start evaluation   ===')
        self.model.eval()
        topk_obj_list, topk_rel_list, topk_triplet_list, cls_matrix_list, edge_feature_list = np.array([]), np.array(
            []), np.array([]), [], []
        sub_scores_list, obj_scores_list, rel_scores_list = [], [], []

        totalStati = {}

        gt_rel_list = []

        for i, items in enumerate(val_loader, 0):
            ''' get data '''
            obj_points, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids, scan_ids = self.data_processing_val(
                items)

            gt_rel_list.append(gt_rel_cls)

            with torch.no_grad():
                top_k_obj, top_k_rel, tok_k_triplet, cls_matrix, sub_scores, obj_scores, rel_scores, scene_accuracy \
                    = self.model.process_val(obj_points, gt_class, descriptor, gt_rel_cls, edge_indices,
                                             batch_ids, use_triplet=True)

            ''' calculate metrics '''
            topk_obj_list = np.concatenate((topk_obj_list, top_k_obj))
            topk_rel_list = np.concatenate((topk_rel_list, top_k_rel))
            topk_triplet_list = np.concatenate((topk_triplet_list, tok_k_triplet))
            if cls_matrix is not None:
                cls_matrix_list.extend(cls_matrix)
                sub_scores_list.extend(sub_scores)
                obj_scores_list.extend(obj_scores)
                rel_scores_list.extend(rel_scores)

            logs = [("Acc@1/obj_cls_acc", (topk_obj_list <= 1).sum() * 100 / len(topk_obj_list)),
                    ("Acc@5/obj_cls_acc", (topk_obj_list <= 5).sum() * 100 / len(topk_obj_list)),
                    ("Acc@10/obj_cls_acc", (topk_obj_list <= 10).sum() * 100 / len(topk_obj_list)),
                    ("Acc@1/rel_cls_acc", (topk_rel_list <= 1).sum() * 100 / len(topk_rel_list)),
                    ("Acc@3/rel_cls_acc", (topk_rel_list <= 3).sum() * 100 / len(topk_rel_list)),
                    ("Acc@5/rel_cls_acc", (topk_rel_list <= 5).sum() * 100 / len(topk_rel_list)),
                    ("Acc@50/triplet_acc", (topk_triplet_list <= 50).sum() * 100 / len(topk_triplet_list)),
                    ("Acc@100/triplet_acc", (topk_triplet_list <= 100).sum() * 100 / len(topk_triplet_list)),
                     ]

            totalStati[scan_ids[0]] = {'rel_acc': scene_accuracy,
                                       'Acc@1/obj_cls_acc': (top_k_obj <= 1).sum() * 100 / len(top_k_obj),
                                       'Acc@50/triplet_acc': (tok_k_triplet <= 50).sum() * 100 / len(tok_k_triplet)}

            # np.save("/home/honsen/honsen/SceneGraph/CVPR2023-VLSAT/statictis_Triplet/totalStati_pre.npy", totalStati,
            #         allow_pickle=True)

            progbar.add(1, values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('Loss')])

        # torch.save(self.model.rel_encoder_3d.gate1list,"/home/honsen/honsen/SceneGraph/baseline/src/gate_param/gate1.pth")
        # torch.save(self.model.rel_encoder_3d.gate2list, "/home/honsen/honsen/SceneGraph/baseline/src/gate_param/gate2.pth")
        # torch.save(self.model.rel_encoder_3d.gate3list, "/home/honsen/honsen/SceneGraph/baseline/src/gate_param/gate3.pth")
        # torch.save(gt_rel_list, "/home/honsen/honsen/SceneGraph/baseline/src/gate_param/gt_rel_list.pth")

        cls_matrix_list = np.stack(cls_matrix_list)
        sub_scores_list = np.stack(sub_scores_list)
        obj_scores_list = np.stack(obj_scores_list)
        rel_scores_list = np.stack(rel_scores_list)
        mean_recall = get_mean_recall(topk_triplet_list, cls_matrix_list)
      

        if self.model.config.EVAL:
            save_path = os.path.join(self.config.PATH, "results", self.model_name, self.exp)
            os.makedirs(save_path, exist_ok=True)
            np.save(os.path.join(save_path, 'topk_pred_list.npy'), topk_rel_list)
            np.save(os.path.join(save_path, 'topk_triplet_list.npy'), topk_triplet_list)
            np.save(os.path.join(save_path, 'cls_matrix_list.npy'), cls_matrix_list)
            np.save(os.path.join(save_path, 'sub_scores_list.npy'), sub_scores_list)
            np.save(os.path.join(save_path, 'obj_scores_list.npy'), obj_scores_list)
            np.save(os.path.join(save_path, 'rel_scores_list.npy'), rel_scores_list)
            f_in = open(os.path.join(save_path, 'result.txt'), 'w')
        else:
            f_in = None

        obj_acc_1 = (topk_obj_list <= 1).sum() * 100 / len(topk_obj_list)
        obj_acc_5 = (topk_obj_list <= 5).sum() * 100 / len(topk_obj_list)
        obj_acc_10 = (topk_obj_list <= 10).sum() * 100 / len(topk_obj_list)
        rel_acc_1 = (topk_rel_list <= 1).sum() * 100 / len(topk_rel_list)
        rel_acc_3 = (topk_rel_list <= 3).sum() * 100 / len(topk_rel_list)
        rel_acc_5 = (topk_rel_list <= 5).sum() * 100 / len(topk_rel_list)
        triplet_acc_50 = (topk_triplet_list <= 50).sum() * 100 / len(topk_triplet_list)
        triplet_acc_100 = (topk_triplet_list <= 100).sum() * 100 / len(topk_triplet_list)

        rel_acc_mean_1, rel_acc_mean_3, rel_acc_mean_5 = self.compute_mean_predicate(cls_matrix_list, topk_rel_list)

        print(f"Eval: 3d obj Acc@1  : {obj_acc_1}", file=f_in)
        print(f"Eval: 3d obj Acc@5  : {obj_acc_5}", file=f_in)
        print(f"Eval: 3d obj Acc@10 : {obj_acc_10}", file=f_in)
        print(f"Eval: 3d rel Acc@1  : {rel_acc_1}", file=f_in)
        print(f"Eval: 3d mean rel Acc@1  : {rel_acc_mean_1}", file=f_in)
        print(f"Eval: 3d rel Acc@3  : {rel_acc_3}", file=f_in)
        print(f"Eval: 3d mean rel Acc@3  : {rel_acc_mean_3}", file=f_in)
        print(f"Eval: 3d rel Acc@5  : {rel_acc_5}", file=f_in)
        print(f"Eval: 3d mean rel Acc@5  : {rel_acc_mean_5}", file=f_in)
        print(f"Eval: 3d triplet Acc@50 : {triplet_acc_50}", file=f_in)
        print(f"Eval: 3d triplet Acc@100 : {triplet_acc_100}", file=f_in)
        print(f"Eval: 3d mean recall@50 : {mean_recall[0]}", file=f_in)
        print(f"Eval: 3d mean recall@100 : {mean_recall[1]}", file=f_in)
    

        if self.model.config.EVAL:
            f_in.close()

        logs = [("Acc@1/obj_cls_acc", obj_acc_1),
                ("Acc@5/obj_cls_acc", obj_acc_5),
                ("Acc@10/obj_cls_acc", obj_acc_10),
                ("Acc@1/rel_cls_acc", rel_acc_1),
                ("Acc@1/rel_cls_acc_mean", rel_acc_mean_1),
                ("Acc@3/rel_cls_acc", rel_acc_3),
                ("Acc@3/rel_cls_acc_mean", rel_acc_mean_3),
                ("Acc@5/rel_cls_acc", rel_acc_5),
                ("Acc@5/rel_cls_acc_mean", rel_acc_mean_5),
                ("Acc@50/triplet_acc", triplet_acc_50),
                ("Acc@100/triplet_acc", triplet_acc_100),
                ("mean_recall@50", mean_recall[0]),
                ("mean_recall@100", mean_recall[1]),
                ]

        self.log(logs, self.model.iteration)
        return mean_recall[0]

    def compute_mean_predicate(self, cls_matrix_list, topk_pred_list):
        cls_dict = {}
        for i in range(26):
            cls_dict[i] = []

        for idx, j in enumerate(cls_matrix_list):
            if j[-1] != -1:
                cls_dict[j[-1]].append(topk_pred_list[idx])

        predicate_mean_1, predicate_mean_3, predicate_mean_5 = [], [], []
        for i in range(26):
            l = len(cls_dict[i])
            if l > 0:
                m_1 = (np.array(cls_dict[i]) <= 1).sum() / len(cls_dict[i])
                m_3 = (np.array(cls_dict[i]) <= 3).sum() / len(cls_dict[i])
                m_5 = (np.array(cls_dict[i]) <= 5).sum() / len(cls_dict[i])
                predicate_mean_1.append(m_1)
                predicate_mean_3.append(m_3)
                predicate_mean_5.append(m_5)

        predicate_mean_1 = np.mean(predicate_mean_1)
        predicate_mean_3 = np.mean(predicate_mean_3)
        predicate_mean_5 = np.mean(predicate_mean_5)

        return predicate_mean_1 * 100, predicate_mean_3 * 100, predicate_mean_5 * 100
