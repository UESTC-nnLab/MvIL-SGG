import json
import os

import numpy as np
import torch
import torch.utils.data as data

from demo import visualPoints_plt
from geo_Process.pointsCompare import compare_point_clouds_from_numpy
from src.utils import op_utils
import random

from src.dataset.relation_aug import random_scale, random_translate, random_jitter, random_dropout, elastic_distortion

def constructSamplelist(root_ScanNet, scene_lists):

    sample_list = []

    for scene in scene_lists:
        cur_path = os.path.join(root_ScanNet,scene)

        rel_labels = np.load(os.path.join(cur_path, "sensorsData/final_rel_labels.npy"),allow_pickle=True).item()

        for key in rel_labels:
            sample_list.append(scene+"$"+key)

    return sample_list

def constructSamplelist1(graph_labels):

    sample_list = []
    for key in graph_labels:
        sample_list.append(key)

    return sample_list

def self_supervise_aug(scene_points, instances):

    import random
    # visualPoints_plt(scene_points)

    elastic_config = {
        'granularity': (4, 4, 4),  # 在x,y,z轴上都分成4段
        'magnitude': (0.2, 0.2, 0.2)  # 各轴位移幅度
    }

    scene_points = elastic_distortion(scene_points, elastic_config['granularity'], elastic_config['magnitude'])

    dropout_ratio = random.uniform(0.3,0.6)

    scene_points, instances = random_dropout(scene_points,instances,dropout_ratio)

    random_angle = random.uniform(60, 300)

    rotated_points = rotate_point_cloud_z(scene_points, random_angle)

    scale_points = random_scale(rotated_points)

    trans_points = random_translate(scale_points)

    jit_points = random_jitter(trans_points)

    # visualPoints_plt(jit_points)

    return jit_points, instances

def self_supervise_aug1(scene_points, instances):

    import random
    # visualPoints_plt(scene_points)

    elastic_config = {
        'granularity': (4, 4, 4),  # 在x,y,z轴上都分成4段
        'magnitude': (0.2, 0.2, 0.2)  # 各轴位移幅度
    }

    instance_ids = np.unique(instances)


    random_angle = random.uniform(60, 300)
    # visualPoints_plt(scene_points)
    rotated_points = rotate_point_cloud_z(scene_points, random_angle)
    # visualPoints_plt(rotated_points)
    scale_points = random_scale(rotated_points)

    trans_points = random_translate(scale_points)
    # visualPoints_plt(trans_points)
    new_points = []
    new_instances =[]

    for in_id in instance_ids:
        current_obj = trans_points[np.where(instances==in_id)]
        # visualPoints_plt(current_obj)
        current_obj_instances = instances[np.where(instances==in_id)]

        # current_obj = elastic_distortion(current_obj, elastic_config['granularity'], elastic_config['magnitude'])

        dropout_ratio = random.uniform(0.55, 0.75)

        current_obj, current_obj_instances = random_dropout(current_obj, current_obj_instances, dropout_ratio)
        # visualPoints_plt(current_obj)
        current_obj = random_jitter(current_obj)
        # visualPoints_plt(current_obj)
        new_points.append(current_obj)
        new_instances.append(current_obj_instances)

    new_scenes_points = np.concatenate(new_points)
    new_scenes_instances = np.concatenate(new_instances)

    # visualPoints_plt(jit_points)

    return new_scenes_points, new_scenes_instances


def rotate_point_cloud_z(points, angle_deg):
    """绕Z轴旋转点云（角度单位为度）"""
    angle_rad = np.deg2rad(angle_deg)
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad),  np.cos(angle_rad), 0],
        [0,                  0,                 1]
    ])
    return np.dot(points, rotation_matrix.T)  # 矩阵乘法

class HonsenDatasetGraph(data.Dataset):
    def __init__(self,
                 # config,
                 split,
                 shuffle_objs,
                 for_train,
                 max_edges=-1):
        assert split in ['train_scannet', 'validation_scannet']
        # self.config = config
        self.for_train = for_train
        self.root_ScanNet = "/home/honsen/tartan/ScanNet/scans"
        self.shuffle_objs = shuffle_objs
        self.max_edges = max_edges
        self.use_descriptor = True
        self.use_data_augmentation = True

        self.num_points = 128
        self.num_points_union = 512

        self.scene_lists = os.listdir(self.root_ScanNet)

        self.graph_labels = np.load("/home/honsen/honsen/SceneGraph/ourPretrained_SGG/data/graph_labels.npy", allow_pickle=True).item()

        self.sample_list = constructSamplelist1(self.graph_labels)#constructSamplelist(self.root_ScanNet, self.scene_lists)
        qwe = len(self.sample_list)
        print()


    def __getitem__(self, index):

        scene_group_id = self.sample_list[index]

        scene_id, group_id = scene_group_id.split("$")

        curScenePath = os.path.join(self.root_ScanNet, scene_id)

        points = np.load(os.path.join(curScenePath,"sensorsData/points.npy"))
        instances = np.load(os.path.join(curScenePath, "sensorsData/instance.npy"))
        final_pesduo_labels = np.load(os.path.join(curScenePath, "sensorsData/final_rel_labels.npy"),allow_pickle=True).item()

        filtered_pesduo_labels = final_pesduo_labels[group_id]

        selected_rels = [key for key in filtered_pesduo_labels]

        clip_feature_labels = self.clip_label(selected_rels, filtered_pesduo_labels)

        obj_points, rel_points, edge_indices, descriptor = \
            self.data_preparation(points.copy(), instances.copy(), self.num_points, self.num_points_union, selected_rels,
                 padding=0.2)


        aug_points, aug_instances = self_supervise_aug1(points.copy(), instances)

        rotated_obj_points, rotated_rel_points, _, rotated_descriptor = \
            self.data_preparation(aug_points, aug_instances, self.num_points, self.num_points_union, selected_rels,
                                  padding=0.2)

        while (len(rel_points) == 0 or filtered_pesduo_labels == []) and self.for_train:
            index = np.random.randint(self.__len__())
            if self.use_descriptor:
                (obj_points, rel_points, descriptor), (rotated_obj_points, rotated_rel_points, rotated_descriptor), edge_indices, clip_feature_labels, scene_group_id = self.__getitem__(index)
            else:
                obj_points, rel_points, gt_rels, edge_indices = self.__getitem__(index)

        if self.use_descriptor: #b0:ori b1:rot b2:edge_idx b3:clip_feat b4:scene_id
             return (obj_points, rel_points, descriptor), (rotated_obj_points, rotated_rel_points, rotated_descriptor), edge_indices, clip_feature_labels, scene_group_id

        return obj_points, rel_points, edge_indices, clip_feature_labels, scene_group_id

    def clip_label(self,selected_rels, filtered_pesduo_labels):

        clip_feature_labels = []

        for rel in selected_rels:
            text_list = filtered_pesduo_labels[rel]
            clip_feature_labels.append(text_list)
        return clip_feature_labels

    def filter_pesduo_labels(self, pesduo_labels, points, instances, insclass_dict):

        filtered_pesduo_labels = {}

        for key in pesduo_labels:
            rel_list = pesduo_labels[key]

            if key not in filtered_pesduo_labels:
                filtered_pesduo_labels[key] = []

            left_v, right_v = int(key.split('_')[0]), int(key.split('_')[1])

            inverse_key = str(right_v)+'_'+str(left_v)
            if inverse_key not in filtered_pesduo_labels:
                filtered_pesduo_labels[inverse_key] = []

            left_class = insclass_dict[left_v]
            right_class = insclass_dict[right_v]

            left_points = points[np.where(instances==left_v)]
            right_points = points[np.where(instances == right_v)]

            flags = [0,0,0] # first index means is bigger, second index means is higher, third index means is connected
            flags = compare_point_clouds_from_numpy(left_points, right_points, flags)
            # inverted_flags = [0 if x == 1 else 1 for x in flags]

            for rel in rel_list:
                for i in range(len(rel)):
                    triplet = rel[i]

                    if len(triplet) != 3:
                        continue

                    relt = triplet[1]

                    if flags[0]==1:
                        if 'part of' in relt or 'small' in relt or 'inside' in relt or 'leaning against' in relt:
                            sentence = right_class+" is "+relt.strip()+" the "+left_class
                            filtered_pesduo_labels[inverse_key].append(sentence)
                            continue
                    if flags[0]==0:
                        if 'bigger' in relt:
                            sentence = right_class+" is "+relt.strip()+" the "+left_class
                            filtered_pesduo_labels[inverse_key].append(sentence)
                            continue
                    if flags[1]==1:
                        if 'lower' in relt or 'supporting' in relt:
                            sentence = right_class+" is "+relt.strip()+" the "+left_class
                            filtered_pesduo_labels[inverse_key].append(sentence)
                            continue
                    if flags[1]==0:
                        if 'standing on' in relt or 'lying on' in relt or 'supported' in relt or 'higher' in relt or 'lying in' in relt:
                            sentence = right_class+" is "+relt.strip()+" the "+left_class
                            filtered_pesduo_labels[inverse_key].append(sentence)
                            continue
                    if flags[2]==0:
                        if 'connected' in relt or 'attached' in relt:
                            sentence = right_class+" is "+relt.strip()+" the "+left_class
                            filtered_pesduo_labels[inverse_key].append(sentence)
                            continue
                    
                    sentence = left_class+" is "+relt.strip()+" the "+right_class
                    filtered_pesduo_labels[key].append(sentence)

            filtered_pesduo_labels[key] = list(set(filtered_pesduo_labels[key]))
            filtered_pesduo_labels[inverse_key] = list(set(filtered_pesduo_labels[inverse_key]))

        remove_key = []
        for key in filtered_pesduo_labels:
            if filtered_pesduo_labels[key] == []:
                remove_key.append(key)

        for key in remove_key:
            del filtered_pesduo_labels[key]

        return filtered_pesduo_labels

    def limit_dict_size(self, input_dict, max_keys=70):
        import random
        current_keys = list(input_dict.keys())
        if len(current_keys) > max_keys:
            excess = len(current_keys) - max_keys
            # 随机选择要删除的键
            keys_to_remove = random.sample(current_keys, excess)
            for key in keys_to_remove:
                del input_dict[key]
        return input_dict

    def getInsClass(self, path):
        scenename = os.path.basename(path)

        filename = scenename + "_vh_clean.aggregation.json"

        insclass_dict = {}

        with open(os.path.join(path, filename), 'r') as f:
            agg_data = json.load(f)

        segGroups = agg_data['segGroups']

        for i in range(len(segGroups)):
            segG = segGroups[i]
            insclass_dict[segG['id']] = segG['label']

        return insclass_dict

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
        return len(self.sample_list)

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

    def relsToInstance(self, selected_rels):

        vertices = set()  # 用集合存储顶点（自动去重）

        for u, v in selected_rels:
            vertices.add(u)
            vertices.add(v)

        # vertices = [int(v) for v in vertices]

        # 转换为排序后的列表（可选）
        vertices = sorted(vertices)

        return vertices

    def subs_rel_idx(self, arrangeidx, selected_rels):

        selected_rels = np.array(selected_rels)

        for key in arrangeidx:
            selected_rels[np.where(selected_rels==key)]=arrangeidx[key]

        return selected_rels

    def data_preparation(self, points, instances, num_points, num_points_union, selected_rels1, scene_id="",
                         padding=0.2,
                        ):  # points: all point coord in the scene, instances: the label of each point
        # all_edge = for_train
        # get instance list

        selected_rels = []

        for i in range(len(selected_rels1)):

            left = int(selected_rels1[i].split('_')[0])
            right = int(selected_rels1[i].split('_')[1])

            selected_rels.append((left,right))

        all_nodes_cur = self.relsToInstance(selected_rels)

        num_objects = len(all_nodes_cur)
        dim_point = points.shape[-1]  # xyz

        instances_box, label_node = dict(), []
        obj_points = torch.zeros([num_objects, num_points, dim_point])
        # obj_points1 = np.zeros([num_objects, num_points, dim_point])
        descriptor = torch.zeros([num_objects, 11])

        arrangeidx = {}

        for i, instance_id in enumerate(all_nodes_cur):
            arrangeidx[instance_id] = i
            # get node point
            obj_pointset = points[np.where(instances == instance_id)[0]]
            min_box = np.min(obj_pointset[:, :3], 0) - padding  # padding object boxes to contain all object points
            max_box = np.max(obj_pointset[:, :3], 0) + padding
            instances_box[instance_id] = (min_box, max_box)  # this two points can decide a 3D boundingbox
            choice = np.random.choice(len(obj_pointset), num_points, replace=True)
            obj_pointset = obj_pointset[choice, :]
            # obj_points1[i] = obj_pointset
            descriptor[i] = op_utils.gen_descriptor(torch.from_numpy(obj_pointset)[:, :3])
            obj_pointset1 = torch.from_numpy(obj_pointset.astype(np.float32))
            obj_pointset1[:, :3] = self.zero_mean(obj_pointset1[:, :3])
            obj_points[i] = obj_pointset1


        rel_points = list()
        for e in range(len(selected_rels)):
            edge = selected_rels[e]
            instance1 = int(edge[0])
            instance2 = int(edge[1])

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

        edge_indices = self.subs_rel_idx(arrangeidx,selected_rels)
        edge_indices = torch.tensor(edge_indices, dtype=torch.long)

        return obj_points, rel_points, edge_indices, descriptor  #,obj_points1
