CUDA_VISIBLE_DEVICES=1 python main.py data/imagenet \
--print-freq 100 -b 256 --worker 8 \
--dataset Cifar10 --arch cifar10_resnet56 --expname cifar10_st_gRDA_pretrained \
--method st_gRDA --prune True \
--pretrained --epoch 200 --prune_steps [30,60,90,120,150,180] --prune_rate 0.8 --lr 0.01 \
--gpu 0