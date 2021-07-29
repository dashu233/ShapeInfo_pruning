CUDA_VISIBLE_DEVICES=6 python main.py data/imagenet \
--print-freq 100 -b 256 --worker 8 --epoch 250\
--dataset Cifar10 --arch cifar10_resnet20 --expname cifar10_scb_pretrained \
--method SCB --prune True \
--pretrained --epoch 200 --prune_steps [30,80,130,180] --prune_rate 0.4 --lr 0.001 \
--gpu 0
