#Structual Margin Pruning

WARN: this method is rely on BN layer, if the model use LN or other normalization layer, this method may not work

To Run Experiment, use the following code

```
python main.py PATH_TO_DATASET \
--print-freq 100 -b 256 --worker 8 \
--dataset YOUR_DATASET --arch YOUR_MODEL_NAME --expname YOU_EXPERIMENT_NAME --pretrained \
--method st_margin --prune True --use_global_criterion True --scaling_norm_criterion False --prune_last True \
--epoch 120 --prune_steps [20,40,60,80,100] --prune_rate 0.5 --lr 0.01 --lr_adjust_steps [20,100] --margin_penalty 5 --wd 1e-5 \
```
# custom model/dataset

To run costum dataset, change `--dataset` to you dataset name and add it into `build_loader.build_data_loader`

To use costum model, change `--arch` to you model name and add it into `custom_model.py`. Read the comments in custom_model.py for more details.

training result will be saved in 'output/YOU_EXPERIMENT_NAME'

# hyperparameter
`--epoch ` total training epoch
`--prune_steps ` prune epoch list for iterative pruning
`--prune_rate ` 1 - final_remain_rate
`--lr ` learning rate
`--lr_adjust_steps ` lr decay epoch, decay rate = 0.1
`--margin_penalty ` the loss weight
`--wd ` weight decay

the other args are some experimental args, please do not change them





