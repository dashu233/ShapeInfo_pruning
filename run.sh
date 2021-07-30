CUDA_VISIBLE_DEVICES=1 python main.py data/imagenet \
--print-freq 100 -b 256 --worker 8 --epoch 1620 \
--dataset Cifar10 --arch cifar10_resnet20 --expname cifar10_scb_pretrained6 \
--method SCB --prune True --scb_slimming_penalty 0.1 --scb_loss_penalty 0.01 \
--pretrained --prune_steps [20,120,220,320,420,520,620,720,820,920,1020,1120,1220,1320,1420,1520] --prune_rate 0.4 --lr 0.01 \
--gpu 0
