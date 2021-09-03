CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py data/imagenet \
--print-freq 100 -b 256 --worker 8 \
--dataset Cifar10 --arch cifar10_resnet56 --expname cifar10_flip_ns_4 \
--method network_slimming --prune True --margin_penalty 5 --use_global_criterion True --hflip False --scaling_norm_criterion False --wd 1e-5 \
--pretrained --epoch 120 --prune_steps [20,40,60,80,100] --prune_rate 0.8 --lr 0.001 --lr_adjust_steps [1000] \
--multiprocessing-distributed
