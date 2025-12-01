import json
import os
import sys
from itertools import product

import numpy as np
import torch
import torch.utils.data as data
import trimesh

from data_processing import compute_weight_occurrences
from demo import visualPoints_plt
from src.utils import op_utils
from utils import define, util, util_data, util_ply
import random
from src.dataset.relation_aug import random_scale, random_translate, random_jitter, random_dropout

def rotate_point_cloud_z(points, angle_deg):
    """绕Z轴旋转点云（角度单位为度）"""
    angle_rad = np.deg2rad(angle_deg)
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad),  np.cos(angle_rad), 0],
        [0,                  0,                 1]
    ])
    return np.dot(points, rotation_matrix.T)  # 矩阵乘法

def self_supervise_aug(scene_points, instances):

    import random
    # visualPoints_plt(scene_points)

    # dropout_ratio = random.uniform(0.3,0.6)
    #
    # scene_points, instances = random_dropout(scene_points,instances,dropout_ratio)
    #
    random_angle = random.uniform(30, 300)

    rotated_points = rotate_point_cloud_z(scene_points, random_angle)
    #
    # scale_points = random_scale(rotated_points)
    #
    # trans_points = random_translate(scale_points)
    #
    # jit_points = random_jitter(trans_points)

    # visualPoints_plt(jit_points)

    return rotated_points, instances

def dataset_loading_3RScan(root: str, pth_selection: str, split: str, class_choice: list = None):
    # read object class
    pth_catfile = os.path.join(pth_selection, 'classes.txt')
    classNames = util.read_txt_to_list(pth_catfile)
    # read relationship class
    pth_relationship = os.path.join(pth_selection, 'relationships.txt')
    util.check_file_exist(pth_relationship)
    relationNames = util.read_relationships(pth_relationship)
    # read relationship json
    selected_scans = set()
    if split == 'train_scans':
        selected_scans = selected_scans.union(util.read_txt_to_list(os.path.join(pth_selection, 'train_scans.txt')))
        with open(os.path.join(root, 'relationships_train.json'), "r") as read_file:
            data = json.load(read_file)
            data = remove_horzi_relationships(data)
    elif split == 'validation_scans':
        selected_scans = selected_scans.union(
            util.read_txt_to_list(os.path.join(pth_selection, 'validation_scans.txt')))
        with open(os.path.join(root, 'relationships_validation.json'), "r") as read_file:
            data = json.load(read_file)
            data = remove_horzi_relationships(data)
    else:
        raise RuntimeError('unknown split type:', split)
    return classNames, relationNames, data, selected_scans

def remove_horzi_relationships(data):
    scans = data['scans']
    for i in range(len(scans)):
        scan = scans[i]
        rels = scan['relationships']
        idxs = []
        for idx, rel in enumerate(rels):
            if 'left' in rel[3] or 'right' in rel[3] or 'behind' in rel[3] or 'front' in rel[3]:
                idxs.append(idx)
        for idx in sorted(idxs,reverse=True):
            rels.pop(idx)
        data['scans'][i]['relationship'] = rels

    return data

def load_mesh(path, label_file, use_rgb, use_normal):
    result = dict()
    if label_file == 'labels.instances.align.annotated.v2.ply' or label_file == 'labels.instances.align.annotated.ply':

        plydata = trimesh.load(os.path.join(path, label_file), process=False)
        points = np.array(plydata.vertices)
        instances = util_ply.read_labels(plydata).flatten()

        if use_rgb:
            rgbs = np.array(plydata.visual.vertex_colors.tolist())[:, :3]
            points = np.concatenate((points, rgbs / 255.0), axis=1)

        if use_normal:
            normal = plydata.vertex_normals[:, :3]
            points = np.concatenate((points, normal), axis=1)

        result['points'] = points
        result['instances'] = instances
    else:
        raise NotImplementedError('')
    return result


class SSGDatasetGraph1(data.Dataset):
    def __init__(self,
                 config,
                 split,
                 multi_rel_outputs,
                 shuffle_objs,
                 use_rgb,
                 use_normal,
                 label_type,
                 for_train,
                 max_edges=-1):
        assert split in ['train_scans', 'validation_scans']
        self.config = config
        self.mconfig = config.dataset
        self.for_train = for_train

        self.root = self.mconfig.root
        self.root_3rscan = define.DATA_PATH
        self.label_type = label_type
        self.scans = []
        self.multi_rel_outputs = multi_rel_outputs
        self.shuffle_objs = shuffle_objs
        self.use_rgb = use_rgb
        self.use_normal = use_normal
        self.max_edges = max_edges
        self.use_descriptor = self.config.MODEL.use_descriptor
        self.use_data_augmentation = self.mconfig.use_data_augmentation
        self.use_2d_feats = self.config.MODEL.use_2d_feats

        if self.mconfig.selection == "":
            self.mconfig.selection = self.root
        self.classNames, self.relationNames, data, selected_scans = \
            dataset_loading_3RScan(self.root, self.mconfig.selection, split)

        # for multi relation output, we just remove off 'None' relationship
        if multi_rel_outputs:
            self.relationNames.pop(0)

        wobjs, wrels, o_obj_cls, o_rel_cls = compute_weight_occurrences.compute(self.classNames, self.relationNames,
                                                                                data, selected_scans, False)
        self.w_cls_obj = torch.from_numpy(np.array(o_obj_cls)).float().to(self.config.DEVICE)
        self.w_cls_rel = torch.from_numpy(np.array(o_rel_cls)).float().to(self.config.DEVICE)

        # for single relation output, we set 'None' relationship weight as 1e-3
        if not multi_rel_outputs:
            self.w_cls_rel[0] = self.w_cls_rel.max() * 10

        self.w_cls_obj = self.w_cls_obj.sum() / (self.w_cls_obj + 1) / self.w_cls_obj.sum()
        self.w_cls_rel = self.w_cls_rel.sum() / (self.w_cls_rel + 1) / self.w_cls_rel.sum()
        self.w_cls_obj /= self.w_cls_obj.max()
        self.w_cls_rel /= self.w_cls_rel.max()

        # print some info
        print('=== {} classes ==='.format(len(self.classNames)))
        for i in range(len(self.classNames)):
            print('|{0:>2d} {1:>20s}'.format(i, self.classNames[i]), end='')
            if self.w_cls_obj is not None:
                print(':{0:>1.3f}|'.format(self.w_cls_obj[i]), end='')
            if (i + 1) % 2 == 0:
                print('')
        print('')
        print('=== {} relationships ==='.format(len(self.relationNames)))
        for i in range(len(self.relationNames)):
            print('|{0:>2d} {1:>20s}'.format(i, self.relationNames[i]), end=' ')
            if self.w_cls_rel is not None:
                print('{0:>1.3f}|'.format(self.w_cls_rel[i]), end='')
            if (i + 1) % 2 == 0:
                print('')
        print('')

        # compile json file
        self.relationship_json, self.objs_json, self.scans = self.read_relationship_json(data, selected_scans)
        print('num of data:', len(self.scans))
        assert (len(self.scans) > 0)

        self.dim_pts = 3
        if self.use_rgb:
            self.dim_pts += 3
        if self.use_normal:
            self.dim_pts += 3

    def __getitem__(self, index):

        scan_id = self.scans[index]
        scan_id_no_split = scan_id.rsplit('_', 1)[0]
        map_instance2labelName = self.objs_json[
            scan_id]  # object list in current scene split : {object_id:"object name"}

        path = os.path.join(self.root_3rscan, scan_id_no_split)

        data = load_mesh(path, self.mconfig.label_file, self.use_rgb, self.use_normal)

        points = data['points']

        instances = data['instances']

        # washmachine = points[np.where(instances==18)]
        # floor = points[np.where(instances == 1)]
        #
        # pairs = np.concatenate((washmachine,floor),axis=0)
        #
        # np.save("/home/honsen/honsen/SceneGraph/ourPretrained_SGG/save_SGG_Tensor/points.npy", points)
        # np.save("/home/honsen/honsen/SceneGraph/ourPretrained_SGG/save_SGG_Tensor/instances.npy", instances)

        aug_points, aug_instances = self_supervise_aug(points, instances)

        rels = self.relationship_json[scan_id]

        # rel_id = []
        # for reli in rels:
        #     rel_id.append(reli[2])
        #
        # rel_id = torch.Tensor(rel_id)

        obj_points, rel_points, gt_rels, gt_class, edge_indices, descriptor,rel_id = \
            self.data_preparation(points, instances, self.mconfig.num_points, self.mconfig.num_points_union,
                                  for_train=self.for_train, instance2labelName=map_instance2labelName,
                                  classNames=self.classNames,  # all class name in the dataset
                                  rel_json=self.relationship_json[scan_id],
                                  relationships=self.relationNames,
                                  multi_rel_outputs=self.multi_rel_outputs,
                                  padding=0.2, num_max_rel=self.max_edges,
                                  shuffle_objs=self.shuffle_objs,
                                  scene_id=scan_id_no_split,
                                  multi_view_root=self.config.multi_view_root)

        # rel_labels = gt_rels.detach().cpu().numpy()
        # rel_labels1 = np.where(rel_labels != 0)

        rot_obj_points, _, _, _, _, rot_descriptor,rel_id = \
            self.data_preparation(aug_points, aug_instances, self.mconfig.num_points, self.mconfig.num_points_union,
                                  for_train=self.for_train, instance2labelName=map_instance2labelName,
                                  classNames=self.classNames,  # all class name in the dataset
                                  rel_json=self.relationship_json[scan_id],
                                  relationships=self.relationNames,
                                  multi_rel_outputs=self.multi_rel_outputs,
                                  padding=0.2, num_max_rel=self.max_edges,
                                  shuffle_objs=self.shuffle_objs,
                                  scene_id=scan_id_no_split,
                                  multi_view_root=self.config.multi_view_root)


        while (len(rel_points) == 0 or gt_rels.sum() == 0) and self.for_train:
            index = np.random.randint(self.__len__())
            if self.use_descriptor:
                if self.use_2d_feats:
                    obj_points, obj_2d_feats, rel_points, gt_class, gt_rels, edge_indices, descriptor = self.__getitem__(
                        index)
                else:
                    obj_points, rel_points, gt_class, gt_rels, edge_indices, descriptor, scan_id = self.__getitem__(
                        index)
            else:
                obj_points, rel_points, gt_class, gt_rels, edge_indices = self.__getitem__(index)

        if self.use_descriptor:
            # print("------------------")
            # print(map_instance2labelName)
            # print(gt_class)
            return obj_points, rel_points, gt_class, gt_rels, edge_indices, descriptor, scan_id

        return obj_points, rel_points, gt_class, gt_rels, edge_indices

    def norm_tensor(self, points):
        assert points.ndim == 2
        assert points.shape[1] == 3
        centroid = torch.mean(points, dim=0)  # N, 3
        points -= centroid  # n, 3, npts
        # furthest_distance = points.pow(2).sum(1).sqrt().max() # find maximum distance for each n -> [n]
        # points /= furthest_distance
        return points

    def zero_mean(self, point):
        mean = torch.mean(point, dim=0)
        point -= mean.unsqueeze(0)
        ''' without norm to 1  '''
        # furthest_distance = point.pow(2).sum(1).sqrt().max() # find maximum distance for each n -> [n]
        # point /= furthest_distance
        return point

    def data_augmentation(self, points):
        # random rotate
        matrix = np.eye(3)
        matrix[0:3, 0:3] = op_utils.rotation_matrix([0, 0, 1], np.random.uniform(0, 2 * np.pi, 1))
        centroid = points[:, :3].mean(0)
        points[:, :3] -= centroid
        points[:, :3] = np.dot(points[:, :3], matrix.T)
        if self.use_normal:
            ofset = 3
            if self.use_rgb:
                ofset += 3
            points[:, ofset:3 + ofset] = np.dot(points[:, ofset:3 + ofset], matrix.T)

        return points

    def __len__(self):
        return len(self.scans)

    def read_relationship_json(self, data, selected_scans: list):
        rel, objs, scans = dict(), dict(), []

        for scan_i in data['scans']:
            if scan_i["scan"] == 'fa79392f-7766-2d5c-869a-f5d6cfb62fc6':
                if self.mconfig.label_file == "labels.instances.align.annotated.v2.ply":
                    '''
                    In the 3RScanV2, the segments on the semseg file and its ply file mismatch. 
                    This causes error in loading data.
                    To verify this, run check_seg.py
                    '''
                    continue
            if scan_i['scan'] not in selected_scans:
                continue

            relationships_i = []
            for relationship in scan_i["relationships"]:
                relationships_i.append(relationship)

            objects_i = {}
            for id, name in scan_i["objects"].items():
                objects_i[int(id)] = name

            rel[scan_i["scan"] + "_" + str(scan_i["split"])] = relationships_i
            objs[scan_i["scan"] + "_" + str(scan_i['split'])] = objects_i
            scans.append(scan_i["scan"] + "_" + str(scan_i["split"]))

        return rel, objs, scans

    def data_preparation(self, points, instances, num_points, num_points_union, scene_id="",
                         # use_rgb, use_normal,
                         for_train=False, instance2labelName=None, classNames=None,
                         rel_json=None, relationships=None, multi_rel_outputs=None,
                         padding=0.2, num_max_rel=-1, shuffle_objs=True, all_edge=False, use_2d_feats=False,
                         multi_view_root=None):  # points: all point coord in the scene, instances: the label of each point
        # all_edge = for_train
        # get instance list
        all_instance = list(np.unique(instances))
        nodes_all = list(instance2labelName.keys())

        if 0 in all_instance:  # remove background
            all_instance.remove(0)

        nodes = []
        for i, instance_id in enumerate(nodes_all):
            if instance_id in all_instance:
                nodes.append(instance_id)

        # get edge (instance pair) list, which is just index, nodes[index] = instance_id
        if all_edge:
            edge_indices = list(product(list(range(len(nodes))), list(range(len(nodes)))))
            # filter out (i,i)
            edge_indices = [i for i in edge_indices if i[0] != i[1]]
        else:
            edge_indices = [(nodes.index(r[0]), nodes.index(r[1])) for r in rel_json if r[0] in nodes and r[1] in nodes]

        rel_ids = [r[2] for r in rel_json if r[0] in nodes and r[1] in nodes]

        num_objects = len(nodes)
        dim_point = points.shape[-1]  # xyz

        instances_box, label_node = dict(), []
        obj_points = torch.zeros([num_objects, num_points, dim_point])
        descriptor = torch.zeros([num_objects, 11])

        for i, instance_id in enumerate(nodes):
            assert instance_id in all_instance, "invalid instance id"
            # get node label name
            instance_name = instance2labelName[instance_id]
            label_node.append(
                classNames.index(instance_name))  # refer to classes.txt, all object was arranged in this file
            # get node point
            obj_pointset = points[np.where(instances == instance_id)[0]]
            min_box = np.min(obj_pointset[:, :3], 0) - padding  # padding object boxes to contain all object points
            max_box = np.max(obj_pointset[:, :3], 0) + padding
            instances_box[instance_id] = (min_box, max_box)  # this two points can decide a 3D boundingbox
            choice = np.random.choice(len(obj_pointset), num_points, replace=True)
            obj_pointset = obj_pointset[choice, :]
            descriptor[i] = op_utils.gen_descriptor(torch.from_numpy(obj_pointset)[:, :3])
            obj_pointset = torch.from_numpy(obj_pointset.astype(np.float32))
            obj_pointset[:, :3] = self.zero_mean(obj_pointset[:, :3])
            obj_points[i] = obj_pointset

        # set gt label for relation
        len_object = len(nodes)
        if multi_rel_outputs:
            adj_matrix_onehot = np.zeros([len_object, len_object, len(relationships)])
        else:
            adj_matrix = np.zeros([len_object, len_object])  # set all to none label.

        for r in rel_json:  # r[object1_index,object2_index,relation_index,relation describ...]
            if r[0] not in nodes or r[1] not in nodes: continue
            assert r[3] in relationships, "invalid relation name"
            r[2] = relationships.index(r[3])  # remap the index of relationships in case of custom relationNames

            if multi_rel_outputs:
                adj_matrix_onehot[nodes.index(r[0]), nodes.index(r[1]), r[2]] = 1
            else:
                adj_matrix[nodes.index(r[0]), nodes.index(r[1])] = r[2]

        # get relation union points
        if multi_rel_outputs:
            adj_matrix_onehot = torch.from_numpy(np.array(adj_matrix_onehot, dtype=np.float32))
            gt_rels = torch.zeros(len(edge_indices), len(relationships), dtype=torch.float)
        else:
            adj_matrix = torch.from_numpy(np.array(adj_matrix, dtype=np.int64))
            gt_rels = torch.zeros(len(edge_indices), dtype=torch.long)

        rel_points = list()
        for e in range(len(edge_indices)):
            edge = edge_indices[e]
            index1 = edge[0]
            index2 = edge[1]
            instance1 = nodes[edge[0]]
            instance2 = nodes[edge[1]]

            if multi_rel_outputs:
                gt_rels[e, :] = adj_matrix_onehot[index1, index2, :]
            else:
                gt_rels[e] = adj_matrix[index1, index2]

            mask1 = (instances == instance1).astype(np.int32) * 1
            mask2 = (instances == instance2).astype(np.int32) * 2
            mask_ = np.expand_dims(mask1 + mask2, 1)
            bbox1 = instances_box[instance1]
            bbox2 = instances_box[instance2]
            min_box = np.minimum(bbox1[0], bbox2[0])
            max_box = np.maximum(bbox1[1], bbox2[1])
            filter_mask = (points[:, 0] > min_box[0]) * (points[:, 0] < max_box[0]) \
                          * (points[:, 1] > min_box[1]) * (points[:, 1] < max_box[1]) \
                          * (points[:, 2] > min_box[2]) * (points[:, 2] < max_box[2])

            # add with context, to distingush the different object's points
            points4d = np.concatenate([points, mask_], 1)

            pointset = points4d[np.where(filter_mask > 0)[0], :]
            choice = np.random.choice(len(pointset), num_points_union, replace=True)
            pointset = pointset[choice, :]
            pointset = torch.from_numpy(pointset.astype(np.float32))
            pointset[:, :3] = self.zero_mean(pointset[:, :3])
            rel_points.append(pointset)

        if len(rel_points) > 0:
            rel_points = torch.stack(rel_points, 0)
        else:
            rel_points = torch.tensor([])

        label_node = torch.from_numpy(np.array(label_node, dtype=np.int64))
        edge_indices = torch.tensor(edge_indices, dtype=torch.long)
        rel_ids = torch.tensor(rel_ids, dtype=torch.long)

        return obj_points, rel_points, gt_rels, label_node, edge_indices, descriptor, rel_ids
