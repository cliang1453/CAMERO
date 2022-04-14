# CAMERO: Consistency Regularized Ensemble of Perturbed Language Models with Weight Sharing

This repo contains our codes for the paper ["CAMERO: Consistency Regularized Ensemble of Perturbed Language Models with Weight Sharing"]() (ACL 2022). We propose a parameter-efficient ensemble approach for large-scale language models based on consistency-regularized perturbed models with weight sharing. Paper link will be coming out soon.

</br>

## Getting Start
1. Pull and run docker </br>
   ```pytorch/pytorch:1.5.1-cuda10.1-cudnn7-devel```
2. Install requirements </br>
   ```pip install -r requirements.txt```

</br>

## Data and Model
1. Download data and pre-trained models following ```download.sh```. Please refer to [this link](https://gluebenchmark.com/) for details on the GLUE benchmark.
2. Preprocess data following ```experiments/glue/prepro.sh```. For the most updated data processing details, please refer to the [mt-dnn repo](https://github.com/namisan/mt-dnn).

</br>

## Training CAMERO
We provide several example scripts for fine-tuning consistency regularized ensemble of perturbed models with weight-sharing. To fine-tune consistency regularized ensemble of perturbed BERT-base models on MNLI dataset, run
```
./scripts/train_mnli.sh GPUID
```

CAMERO has several important hyper-parameters that you can play with:
- ```--n_models```: The number of models, e.g., 2 and 4.
- ```--teaching_type```: The types consistency regularization.
   - ```"ensemble"```: the consistency loss is computed based on the average distance between the ensemble of all models' logits and individual models' logits.
   - ```"pairwise"```: the consistency loss is computed based on the average distance between every two models' logits.
- ```--pert_type```: The types of perturbation added to the models' hidden representations.
   - ```"dropout"```: dropout ([Srivastava et al., 2014](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)).
   - ```"adv"```: virtual adversarial perturbation ([Jiang et al., 2019](https://arxiv.org/pdf/1911.03437.pdf)).
   - ```"r3f"```: random noise perturbation ([Aghajanyan et al., 2020](https://arxiv.org/pdf/2008.03156.pdf)). </br>
   Using ```"dropout"``` is often sufficient to get good results. ```"adv"``` may lead to better results in certain tasks but require longer training time.
- ```--kd_alpha```: The weight of consistency loss. Sensitive to the type of tasks.

A few other notices:
- To fine-tune a RoBERTa model, download the model checkpoint following ```download.sh```, set ```--init_checkpoint``` to the checkpoint path and set ```--encoder_type``` to ```2```. Other supported models are listed in ```pretrained_models.py```.
- To fine-tune models on other tasks, set ```--train_datasets``` and ```--test_datasets``` to the corresponding task names.
- All models share their encoder weights. The final saved checkpoint is a single encoder with ```n_models``` classification heads.

</br>

## Citation

Coming out soon.
```
```

</br>

## Contact Information
For help or issues related to this package, please submit a GitHub issue. For personal questions related to this paper, please contact Chen Liang (cliang73@gatech.edu).
