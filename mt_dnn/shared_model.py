# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
import copy
import imp
import sys
import torch
import tasks
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import *
from data_utils.utils import AverageMeter
from mt_dnn.loss import LOSS_REGISTRY
from mt_dnn.shared_matcher import MultiClsSANBertNetwork
from mt_dnn.perturbation import SmartPerturbation
from mt_dnn.loss import *
from mt_dnn.optim import AdamaxW
from data_utils.task_def import TaskType, EncoderModelType
from experiments.exp_def import TaskDef
from data_utils.my_statics import DUMPY_STRING_FOR_EMPTY_ANS


logger = logging.getLogger(__name__)


class MultiClsMTDNNModel(object):
    def __init__(self, opt, device=None, state_dict=None, num_train_step=-1):
        self.config = opt
        self.updates = (
            state_dict["updates"] if state_dict and "updates" in state_dict else 0
        )
        self.local_updates = 0
        self.device = device

        # CAMERO args #
        self.n_models = self.config['n_models']
        self.pert_type = self.config['pert_type']
        self.teaching_type = self.config['teaching_type']
        ###############
    
        self.train_loss = AverageMeter()
        self.adv_loss = AverageMeter()
        self.emb_val = AverageMeter()
        self.eff_perturb = AverageMeter()
        self.initial_from_local = True if state_dict else False
        model = MultiClsSANBertNetwork(opt, initial_from_local=self.initial_from_local)
        self.total_param = sum(
            [p.nelement() for p in model.parameters() if p.requires_grad]
        )
        if opt["cuda"]:
            model = model.to(self.device)
        self.network = model
        if state_dict:
            missing_keys, unexpected_keys = self.network.load_state_dict(
                state_dict["state"], strict=False
            )

        optimizer_parameters = self._get_param_groups()
        self._setup_optim(optimizer_parameters, state_dict, num_train_step)
        self.optimizer.zero_grad()

        # if self.config["local_rank"] not in [-1, 0]:
        #     torch.distributed.barrier()

        if self.config["local_rank"] != -1:
            self.mnetwork = torch.nn.parallel.DistributedDataParallel(
                self.network,
                device_ids=[self.config["local_rank"]],
                output_device=self.config["local_rank"],
                find_unused_parameters=True,
            )
        elif self.config["multi_gpu_on"]:
            self.mnetwork = nn.DataParallel(self.network)
        else:
            self.mnetwork = self.network
        self._setup_lossmap(self.config)
        self._setup_kd_lossmap(self.config)
        self._setup_adv_lossmap(self.config)
        self._setup_adv_training(self.config)
        self._setup_tokenizer()

    def _setup_adv_training(self, config):

        self.adv_teacher = None
        if config.get("pert_type", "none") in ["adv", "r3f"]:
            self.adv_teacher = SmartPerturbation(
                config["adv_epsilon"],
                config["multi_gpu_on"],
                config["adv_step_size"],
                config["adv_noise_var"],
                config["adv_p_norm"],
                config["adv_k"],
                config["fp16"],
                config["encoder_type"],
                loss_map=self.adv_task_loss_criterion,
                norm_level=config["adv_norm_level"],
            )

    def _get_param_groups(self):
        ''' copied from https://github.com/namisan/mt-dnn/blob/master/mt_dnn/model.py'''

        no_decay = ["bias", "gamma", "beta", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p
                    for n, p in self.network.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in self.network.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        return optimizer_parameters

    def _setup_optim(self, optimizer_parameters, state_dict=None, num_train_step=-1):
        ''' copied from https://github.com/namisan/mt-dnn/blob/master/mt_dnn/model.py'''

        if self.config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(optimizer_parameters, self.config['learning_rate'],
                                       weight_decay=self.config['weight_decay'])
        elif self.config['optimizer'] == 'adamax':
            self.optimizer = AdamaxW(optimizer_parameters,
                                    lr=self.config['learning_rate'],
                                    weight_decay=self.config['weight_decay'])
        elif self.config['optimizer'] == 'adam':
            self.optimizer = optim.AdamW(optimizer_parameters,
                                    lr=self.config['learning_rate'],
                                    weight_decay=self.config['weight_decay'])
        else:
            raise RuntimeError('Unsupported optimizer: %s' % opt['optimizer'])

        if state_dict and 'optimizer' in state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])


        if state_dict and "optimizer" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer"])

        if self.config["fp16"]:
            try:
                from apex import amp
                global amp
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
                )
            model, optimizer = amp.initialize(
                self.network, self.optimizer, opt_level=self.config["fp16_opt_level"]
            )
            self.network = model
            self.optimizer = optimizer

        # # set up scheduler
        self.scheduler = None
        scheduler_type = self.config['scheduler_type']
        warmup_steps = self.config['warmup'] * num_train_step
        if scheduler_type == 3:
            from transformers import get_polynomial_decay_schedule_with_warmup
            self.scheduler = get_polynomial_decay_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_train_step
                )
        if scheduler_type == 2:
            from transformers import get_constant_schedule_with_warmup
            self.scheduler = get_constant_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps
                )
        elif scheduler_type == 1:
            from transformers import get_cosine_schedule_with_warmup
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_train_step
                )
        else:
            from transformers import get_linear_schedule_with_warmup
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_train_step
                )

    def _setup_lossmap(self, config):
        ''' copied from https://github.com/namisan/mt-dnn/blob/master/mt_dnn/model.py'''

        task_def_list: List[TaskDef] = config["task_def_list"]
        self.task_loss_criterion = []
        for idx, task_def in enumerate(task_def_list):
            cs = task_def.loss
            lc = LOSS_REGISTRY[cs](name="Loss func of task {}: {}".format(idx, cs))
            self.task_loss_criterion.append(lc)

    def _setup_kd_lossmap(self, config):

        task_def_list: List[TaskDef] = config["task_def_list"]
        self.kd_task_loss_criterion = []
        for idx, task_def in enumerate(task_def_list):
            cs = task_def.kd_loss
            assert cs is not None
            lc = LOSS_REGISTRY[cs](
                name="KD Loss func of task {}: {}".format(idx, cs)
            )
            self.kd_task_loss_criterion.append(lc)

    def _setup_adv_lossmap(self, config):

        task_def_list: List[TaskDef] = config["task_def_list"]
        self.adv_task_loss_criterion = []
        if config.get("pert_type", "none") in ["adv", "r3f"]:
            for idx, task_def in enumerate(task_def_list):
                cs = task_def.adv_loss
                assert cs is not None
                lc = LOSS_REGISTRY[cs](
                    name="Adv Loss func of task {}: {}".format(idx, cs)
                )
                self.adv_task_loss_criterion.append(lc)
    
    def _setup_tokenizer(self):
        ''' copied from https://github.com/namisan/mt-dnn/blob/master/mt_dnn/model.py'''

        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config["init_checkpoint"],
                cache_dir=self.config["transformer_cache"],
            )
        except:
            self.tokenizer = None

    def _to_cuda(self, tensor):
        ''' copied from https://github.com/namisan/mt-dnn/blob/master/mt_dnn/model.py'''
        
        if tensor is None:
            return tensor

        if isinstance(tensor, list) or isinstance(tensor, tuple):
            # y = [e.cuda(non_blocking=True) for e in tensor]
            y = [e.to(self.device) for e in tensor]
            for e in y:
                e.requires_grad = False
        else:
            # y = tensor.cuda(non_blocking=True)
            y = tensor.to(self.device)
            y.requires_grad = False
        return y

    def update(self, batch_meta, batch_data):

        self.network.train()
        y = batch_data[batch_meta["label"]]
        y = self._to_cuda(y) if self.config["cuda"] else y
        if batch_meta["task_def"]["task_type"] == TaskType.SeqenceGeneration:
            seq_length = y.size(1)
            y = y.view(-1)

        task_id = batch_meta["task_id"]
        inputs = batch_data[: batch_meta["input_len"]]
        if self.network.encoder_type == EncoderModelType.ROBERTA:
            inputs[1] = None
        if len(inputs) == 3:
            inputs.append(None)
            inputs.append(None)
        inputs.append(task_id)
        if "y_token_id" in batch_meta:
            inputs.append(batch_data[batch_meta["y_token_id"]])
        weight = None
        if self.config.get("weighted_on", False):
            if self.config["cuda"]:
                weight = batch_data[batch_meta["factor"]].cuda(non_blocking=True)
            else:
                weight = batch_data[batch_meta["factor"]]

        # conduct forward passes to get perturbed logits of all models
        pert_logits_list = []
        if self.pert_type == "none": # no perturbation
            pert_logits_list = self.mnetwork(*inputs, group_id=[j for j in range(self.n_models)])
            
        elif self.pert_type == "dropout": # dropout perturbation
            for j in range(self.n_models):
                logits = self.mnetwork(*inputs, group_id=j)
                pert_logits_list.append(logits)

        elif self.pert_type == "adv": # adversarial perturbation
            logits_list = []
            task_type = batch_meta['task_def']['task_type']
            for j in range(self.n_models):
                logits = self.mnetwork(*inputs, group_id=j)
                logits_list.append(logits)

            for j in range(self.n_models):
                pert_inputs = [self.mnetwork, logits_list[j]] + inputs
                pert_logits = self.adv_teacher.get_adv_perturbed_logits(*pert_inputs, task_type=task_type, pairwise=batch_meta.get('pairwise_size', 1), group_id=j)
                pert_logits_list.append(pert_logits)

        elif self.pert_type == "r3f": # random perturbation
            pert_embed_list = []
            task_type = batch_meta['task_def']['task_type']
            for j in range(self.n_models):
                pert_inputs = [self.mnetwork] + inputs
                pert_embed = self.adv_teacher.get_random_perturbed_embed(*pert_inputs, task_type=task_type, group_id=j)
                pert_embed_list.append(pert_embed)

            for j in range(self.n_models):
                pert_logits = self.mnetwork(*inputs, fwd_type=2, embed=pert_embed_list[j], group_id=j)
                pert_logits_list.append(pert_logits)
        
        else:
            print("This perturbation type is currently not supported.")
            exit()

        # compute averaged training loss of n models
        loss = 0
        if self.task_loss_criterion[task_id] and (y is not None):
            loss_criterion = self.task_loss_criterion[task_id]
            for j in range(self.n_models):
                if (
                    isinstance(loss_criterion, RankCeCriterion)
                    and batch_meta["pairwise_size"] > 1
                ):
                    # reshape the logits for ranking.
                    loss += self.task_loss_criterion[task_id](
                        pert_logits_list[j],
                        y,
                        weight,
                        ignore_index=-1,
                        pairwise_size=batch_meta["pairwise_size"],
                    )
                elif batch_meta["task_def"]["task_type"] == TaskType.SeqenceGeneration:
                    weight = (
                        (
                            1.0
                            / torch.sum(
                                (y > -1).float().view(-1, seq_length), 1, keepdim=True
                            )
                        )
                        .repeat(1, seq_length)
                        .view(-1)
                    )
                    loss += self.task_loss_criterion[task_id](
                        pert_logits_list[j], y, weight, ignore_index=-1
                    )
                else:
                    loss += self.task_loss_criterion[task_id](
                        pert_logits_list[j], y, weight, ignore_index=-1
                    )
            loss /= self.n_models
    
        # compute consistency loss among n models
        kd_loss = 0
        if self.kd_task_loss_criterion[task_id]:
            kd_lc = self.kd_task_loss_criterion[task_id]
            if self.teaching_type == "pairwise":
                order = torch.randperm(self.n_models)
                for j in range(self.n_models):
                    kd_loss += (
                        kd_lc(pert_logits_list[order[j]], pert_logits_list[order[(j+1)%self.n_models]], weight=None, ignore_index=-1) if kd_lc else 0
                    )
            elif self.teaching_type == "ensemble":
                logits_mean = torch.stack(pert_logits_list).mean(axis=0)
                for j in range(self.n_models):
                    kd_loss += (
                        kd_lc(pert_logits_list[j], logits_mean, weight=None, ignore_index=-1) if kd_lc else 0
                    )
            else:
                print("This teaching type is currently not supported.")
                exit()
            kd_loss /= self.n_models
            loss = loss + self.config["kd_alpha"] * kd_loss

        batch_size = batch_data[batch_meta["token_id"]].size(0)
        # rescale loss as dynamic batching
        if self.config["bin_on"]:
            loss = loss * (1.0 * batch_size / self.config["batch_size"])
        if self.config["local_rank"] != -1:
            # print('Rank ', self.config['local_rank'], ' loss ', loss)
            copied_loss = copy.deepcopy(loss.data)
            torch.distributed.all_reduce(copied_loss)
            copied_loss = copied_loss / self.config["world_size"]
            self.train_loss.update(copied_loss.item(), batch_size)
        else:
            self.train_loss.update(loss.item(), batch_size)

        # scale loss
        loss = loss / self.config.get("grad_accumulation_step", 1)
        if self.config["fp16"]:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        self.local_updates += 1
        if self.local_updates % self.config.get("grad_accumulation_step", 1) == 0:
            if self.config["global_grad_clipping"] > 0:
                if self.config["fp16"]:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(self.optimizer),
                        self.config["global_grad_clipping"],
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.network.parameters(), self.config["global_grad_clipping"]
                    )
            self.updates += 1
            # reset number of the grad accumulation
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.scheduler:
                self.scheduler.step()

    def encode(self, batch_meta, batch_data):
        ''' copied from https://github.com/namisan/mt-dnn/blob/master/mt_dnn/model.py'''

        self.network.eval()
        inputs = batch_data[:3]
        sequence_output = self.network.encode(*inputs)[0]
        return sequence_output

    # TODO: similar as function extract, preserve since it is used by extractor.py
    # will remove after migrating to transformers package
    def extract(self, batch_meta, batch_data):
        ''' copied from https://github.com/namisan/mt-dnn/blob/master/mt_dnn/model.py'''

        self.network.eval()
        # 'token_id': 0; 'segment_id': 1; 'mask': 2
        inputs = batch_data[:3]
        all_encoder_layers, pooled_output = self.mnetwork.bert(*inputs)
        return all_encoder_layers, pooled_output

    def predict_logits(self, batch_meta, batch_data, group_id=0):
        '''predict logits for the j-th model.'''

        self.network.eval()
        task_id = batch_meta["task_id"]
        task_def = TaskDef.from_dict(batch_meta["task_def"])
        task_type = task_def.task_type
        task_obj = tasks.get_task_obj(task_def)
        inputs = batch_data[: batch_meta["input_len"]]
        if len(inputs) == 3:
            inputs.append(None)
            inputs.append(None)
        inputs.append(task_id)
        if task_type == TaskType.SeqenceGeneration:
            # y_idx, #3 -> gen
            inputs.append(None)
            inputs.append(3)
        
        score = self.mnetwork(*inputs, group_id=group_id)
        return score

    def predict(self, batch_meta, batch_data, group_id=0):
        '''evaluate the j-th model.'''

        self.network.eval()
        task_id = batch_meta["task_id"]
        task_def = TaskDef.from_dict(batch_meta["task_def"])
        task_type = task_def.task_type
        task_obj = tasks.get_task_obj(task_def)
        inputs = batch_data[: batch_meta["input_len"]]
        if len(inputs) == 3:
            inputs.append(None)
            inputs.append(None)
        inputs.append(task_id)
        if task_type == TaskType.SeqenceGeneration:
            # y_idx, #3 -> gen
            inputs.append(None)
            inputs.append(3)

        score = self.mnetwork(*inputs, group_id=group_id)
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

    def save(self, filename):
        ''' copied from https://github.com/namisan/mt-dnn/blob/master/mt_dnn/model.py'''

        if isinstance(self.mnetwork, torch.nn.parallel.DistributedDataParallel):
            model = self.mnetwork.module
        else:
            model = self.network
        #network_state = dict([(k, v.cpu()) for k, v in self.network.state_dict().items()])
        network_state = dict([(k, v.cpu()) for k, v in model.state_dict().items()])
        params = {
            'state': network_state,
            'optimizer': self.optimizer.state_dict(),
            'config': self.config,
        }
        torch.save(params, filename)
        logger.info('model saved to {}'.format(filename))

    def load(self, checkpoint):
        ''' copied from https://github.com/namisan/mt-dnn/blob/master/mt_dnn/model.py'''

        model_state_dict = torch.load(checkpoint)
        if "state" in model_state_dict:
            self.network.load_state_dict(model_state_dict["state"], strict=False)
        if "optimizer" in model_state_dict:
            self.optimizer.load_state_dict(model_state_dict["optimizer"])
        if "config" in model_state_dict:
            self.config.update(model_state_dict["config"])

    def cuda(self):
        ''' copied from https://github.com/namisan/mt-dnn/blob/master/mt_dnn/model.py'''

        self.network.cuda()
