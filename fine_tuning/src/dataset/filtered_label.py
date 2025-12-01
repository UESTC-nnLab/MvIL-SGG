from geo_Process.pointsCompare import compare_point_clouds_from_numpy
import numpy as np
import os
import json
import random
def getInsClass(path):
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

def split_dict_into_chunks(original_dict, min_chunk=5, max_chunk=15):
    keys = list(original_dict.keys())
    total_keys = len(keys)
    chunks = []
    
    # 随机拆分键，确保每块大小在 min_chunk 和 max_chunk 之间
    remaining = total_keys
    while remaining > 0:
        # 计算当前块的大小（不能超过剩余键数，也不能超过 max_chunk）
        chunk_size = random.randint(min_chunk, min(max_chunk, remaining))
        if remaining - chunk_size < min_chunk and remaining - chunk_size > 0:
            # 如果剩下的键数不足以构成一个最小块，则调整当前块的大小
            chunk_size = remaining
        chunks.append(chunk_size)
        remaining -= chunk_size
    
    # 打乱键的顺序（可选，确保随机性）
    random.shuffle(keys)
    
    # 构建二级字典
    nested_dict = {}
    start = 0
    for i, chunk_size in enumerate(chunks):
        chunk_keys = keys[start : start + chunk_size]
        chunk_dict = {k: original_dict[k] for k in chunk_keys}
        nested_dict[f"group_{i+1}"] = chunk_dict
        start += chunk_size
    
    return nested_dict

def filter_label():
    from tqdm import tqdm
    datapath = "/home/honsen/tartan/ScanNet/scans"

    scene_list = os.listdir(datapath)

    for scene in tqdm(scene_list):
        pesduo_labels = np.load(os.path.join(datapath, scene, "sensorsData/rel_label.npy"), allow_pickle=True).item()
        points = np.load(os.path.join(datapath, scene, "sensorsData/points.npy"))
        instances = np.load(os.path.join(datapath, scene, "sensorsData/instance.npy"))
        insclass_dict = getInsClass(os.path.join(datapath, scene))
        filtered_pesduo_labels = filter_pesduo_labels(pesduo_labels, points, instances, insclass_dict)

        final_pesduo_labels = split_dict_into_chunks(filtered_pesduo_labels)

        np.save(os.path.join(datapath, scene, "sensorsData/final_rel_labels.npy"),final_pesduo_labels, allow_pickle=True)

def filter_pesduo_labels(pesduo_labels, points, instances, insclass_dict):

    filtered_pesduo_labels = {}

    for key in pesduo_labels:
        rel_list = pesduo_labels[key]

        if key not in filtered_pesduo_labels:
            filtered_pesduo_labels[key] = []

        left_v, right_v = int(key.split('_')[0]), int(key.split('_')[1])

        inverse_key = str(right_v) + '_' + str(left_v)
        if inverse_key not in filtered_pesduo_labels:
            filtered_pesduo_labels[inverse_key] = []

        left_class = insclass_dict[left_v]
        right_class = insclass_dict[right_v]


        left_points = points[np.where(instances == left_v)]
        right_points = points[np.where(instances == right_v)]

        flags = [0, 0, 0]  # first index means is bigger, second index means is higher, third index means is connected
        flags = compare_point_clouds_from_numpy(left_points, right_points, flags)
        # inverted_flags = [0 if x == 1 else 1 for x in flags]

        for rel in rel_list:
            for i in range(len(rel)):
                triplet = rel[i]

                if len(triplet) != 3:
                    continue

                relt = triplet[1]
                left_obj = triplet[0]
                right_obj = triplet[2]

                if left_class not in left_obj and left_class not in right_obj:
                    continue

                if right_class not in left_obj and right_class not in right_obj:
                    continue

                if ("attached" in relt or "hanging" in relt or "supported" in relt) and "wall" in left_obj:
                    sentence = right_class + " is " + relt.strip() + " the " + left_class
                    filtered_pesduo_labels[inverse_key].append(sentence)
                    continue

                if left_obj != right_obj and "same" in relt:
                    continue

                if flags[0] == 1:
                    if 'part of' in relt or 'small' in relt or 'inside' in relt or 'leaning against' in relt:
                        sentence = right_class + " is " + relt.strip() + " the " + left_class
                        filtered_pesduo_labels[inverse_key].append(sentence)
                        continue
                if flags[0] == 0:
                    if 'bigger' in relt:
                        sentence = right_class + " is " + relt.strip() + " the " + left_class
                        filtered_pesduo_labels[inverse_key].append(sentence)
                        continue
                if flags[1] == 1:
                    if 'lower' in relt or 'supporting' in relt:
                        sentence = right_class + " is " + relt.strip() + " the " + left_class
                        filtered_pesduo_labels[inverse_key].append(sentence)
                        continue
                if flags[1] == 0:
                    if 'standing' in relt or 'lying on' in relt or 'supported' in relt or 'higher' in relt or 'lying in' in relt:
                        sentence = right_class + " is " + relt.strip() + " the " + left_class
                        filtered_pesduo_labels[inverse_key].append(sentence)
                        continue
                if flags[2] == 0:
                    if 'connected' in relt or 'attached' in relt:
                        sentence = right_class + " is " + relt.strip() + " the " + left_class
                        filtered_pesduo_labels[inverse_key].append(sentence)
                        continue

                sentence = left_class + " is " + relt.strip() + " the " + right_class
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

if __name__ =="__main__":

    filter_label()
