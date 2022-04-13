# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
import enum
from numpy.lib.arraysetops import isin
from numpy.lib.function_base import insert
from data_utils.metrics import calc_metrics
from mt_dnn.batcher import Collater
from data_utils.task_def import TaskType
from data_utils.utils_qa import postprocess_qa_predictions
from copy import deepcopy
import numpy as np
import torch
from tqdm import tqdm
from experiments.exp_def import TaskDef
import tasks
import torch.nn.functional as F

def extract_encoding(model, data, use_cuda=True):
    ''' copied from https://github.com/namisan/mt-dnn/blob/master/mt_dnn/inference.py'''
    if use_cuda:
        model.cuda()
    sequence_outputs = []
    max_seq_len = 0
    for idx, (batch_info, batch_data) in enumerate(data):
        batch_info, batch_data = Collater.patch_data(use_cuda, batch_info, batch_data)
        sequence_output = model.encode(batch_info, batch_data)
        sequence_outputs.append(sequence_output)
        max_seq_len = max(max_seq_len, sequence_output.shape[1])

    new_sequence_outputs = []
    for sequence_output in sequence_outputs:
        new_sequence_output = torch.zeros(
            sequence_output.shape[0], max_seq_len, sequence_output.shape[2]
        )
        new_sequence_output[:, : sequence_output.shape[1], :] = sequence_output
        new_sequence_outputs.append(new_sequence_output)

    return torch.cat(new_sequence_outputs)

def reduce_multirc(uids, predictions, golds):
    ''' copied from https://github.com/namisan/mt-dnn/blob/master/mt_dnn/inference.py'''
    assert len(uids) == len(predictions)
    assert len(uids) == len(golds)
    from collections import defaultdict

    predict_map = defaultdict(list)
    gold_map = defaultdict(list)
    for idx, uid in enumerate(uids):
        blocks = uid.split("_")
        assert len(blocks) == 3
        nuid = "_".join(blocks[:-1])
        predict_map[uid].append(predictions[idx])
        gold_map[uid].append(golds[idx])
    return predict_map, gold_map

def merge(src, tgt):
    def _mg(src, tgt):
        if isinstance(src, dict):
            for k, v in src.items():
                if k in tgt:
                    tgt[k] = _mg(v, tgt[k])
                else:
                    tgt[k] = v
        elif isinstance(src, list):
            tgt.extend(src)
        elif isinstance(src, tuple):
            if isinstance(src[0], list):
                for i, k in enumerate(src):
                    tgt[i].extend(src[i])
            else:
                tgt.extend(src)
        else:
            tgt = src
        return tgt

    if tgt is None or len(tgt) == 0:
        tgt = deepcopy(src)
        return tgt
    else:
        return _mg(src, tgt)

def eval_model(
    model,
    data,
    metric_meta,
    device,
    with_label=True,
    label_mapper=None,
    task_type=TaskType.Classification,
    group_id=0,
):
    
    predictions = []
    golds = []
    scores = []
    ids = []
    metrics = {}
    for (batch_info, batch_data) in tqdm(data, total=len(data)):
        batch_info, batch_data = Collater.patch_data(device, batch_info, batch_data)
        
        if isinstance(group_id, int):
            score, pred, gold = model.predict(batch_info, batch_data, group_id=group_id)
        elif isinstance(group_id, list):
            if task_type == TaskType.Span:
                start_scores_list, end_scores_list = [], []
                for j in group_id:
                    start_scores, end_scores = model.predict_logits(batch_info, batch_data, group_id=j)
                    start_scores_list.append(start_scores)
                    end_scores_list.append(end_scores)
                start_scores_mean = torch.stack(start_scores_list).mean(axis=0)
                end_scores_mean = torch.stack(end_scores_list).mean(axis=0)
                logits_mean = (start_scores_mean, end_scores_mean)
            else:
                logits_list = []
                for j in group_id:
                    logits = model.predict_logits(batch_info, batch_data, group_id=j)
                    logits_list.append(logits)
                logits_mean = torch.stack(logits_list).mean(axis=0)
            score, pred, gold = predict_from_logits(logits_mean, batch_info, batch_data)
        else:
            print("This type of group_id is not supported.")
            exit()

        scores = merge(score, scores)
        golds = merge(gold, golds)
        predictions = merge(pred, predictions)
        ids = merge(batch_info["uids"], ids)
    
    if task_type == TaskType.Span:
        predictions, golds = postprocess_qa_predictions(
            golds, scores, version_2_with_negative=False
        )
    elif task_type == TaskType.SpanYN:
        predictions, golds = postprocess_qa_predictions(
            golds, scores, version_2_with_negative=True
        )

    if with_label:
        metrics = calc_metrics(metric_meta, golds, predictions, scores, label_mapper)
    return metrics, predictions, scores, golds, ids

def predict_from_logits(score, batch_meta, batch_data):

    task_id = batch_meta["task_id"]
    task_def = TaskDef.from_dict(batch_meta["task_def"])
    task_type = task_def.task_type
    task_obj = tasks.get_task_obj(task_def)

    if task_obj is not None:
        score, predict = task_obj.test_predict(score)
    elif task_type == TaskType.Ranking:
        score = score.contiguous().view(-1, batch_meta["pairwise_size"])
        assert task_type == TaskType.Ranking
        score = F.softmax(score, dim=1)
        score = score.data.cpu()
        score = score.numpy()
        predict = np.zeros(score.shape, dtype=int)
        positive = np.argmax(score, axis=1)
        for idx, pos in enumerate(positive):
            predict[idx, pos] = 1
        predict = predict.reshape(-1).tolist()
        score = score.reshape(-1).tolist()
        return score, predict, batch_meta["true_label"]
    elif task_type == TaskType.SeqenceLabeling:
        mask = batch_data[batch_meta["mask"]]
        score = score.contiguous()
        score = score.data.cpu()
        score = score.numpy()
        predict = np.argmax(score, axis=1).reshape(mask.size()).tolist()
        valied_lenght = mask.sum(1).tolist()
        final_predict = []
        for idx, p in enumerate(predict):
            final_predict.append(p[: valied_lenght[idx]])
        score = score.reshape(-1).tolist()
        return score, final_predict, batch_meta["label"]
    elif task_type == TaskType.Span or task_type == TaskType.SpanYN:
        predictions = []
        features = []
        for idx, offset in enumerate(batch_meta["offset_mapping"]):
            token_is_max_context = (
                batch_meta["token_is_max_context"][idx]
                if batch_meta.get("token_is_max_context", None)
                else None
            )
            sample_id = batch_meta["uids"][idx]
            if "label" in batch_meta:
                feature = {
                    "offset_mapping": offset,
                    "token_is_max_context": token_is_max_context,
                    "uid": sample_id,
                    "context": batch_meta["context"][idx],
                    "answer": batch_meta["answer"][idx],
                    "label": batch_meta["label"][idx],
                }
            else:
                feature = {
                    "offset_mapping": offset,
                    "token_is_max_context": token_is_max_context,
                    "uid": sample_id,
                    "context": batch_meta["context"][idx],
                    "answer": batch_meta["answer"][idx],
                }
            if "null_ans_index" in batch_meta:
                feature["null_ans_index"] = batch_meta["null_ans_index"]
            features.append(feature)
        start, end = score
        start = start.contiguous()
        start = start.data.cpu()
        start = start.numpy().tolist()
        end = end.contiguous()
        end = end.data.cpu()
        end = end.numpy().tolist()
        return (start, end), predictions, features
    elif task_type == TaskType.SeqenceGeneration:
        predicts = self.tokenizer.batch_decode(score, skip_special_tokens=True)
        predictions = {}
        golds = {}
        for idx, predict in enumerate(predicts):
            sample_id = batch_meta["uids"][idx]
            answer = batch_meta["answer"][idx]
            predict = predict.strip()
            if predict == DUMPY_STRING_FOR_EMPTY_ANS:
                predict = ""
            predictions[sample_id] = predict
            golds[sample_id] = answer
        score = score.contiguous()
        score = score.data.cpu()
        score = score.numpy().tolist()
        return score, predictions, golds
    else:
        raise ValueError("Unknown task_type: %s" % task_type)
    return score, predict, batch_meta["label"]
