# Cell segmentation using convolutional neural network

Documentation
main.py: Start file for training the network. Specify path to dataset, tensorboard log directory etc here.
dataloader.py: Simple basic pytorch dataloader for DSB2018.
distance_loss.py : Loss function class, as defined in the paper.
train.py: Trainer file. Code to load and save checkpoints in accordance with loss and learning rate.
load_save_model.py: Boilerplate assisting code for the Trainer class.Used for loading and saving models.
predict.py: Script for test set prediction. Specify path to test set and pretrained weights here. User can also change the probability threshold for NMS.
metric.py: Unofficial evaluation metric script for calculating average precision of IoUs.
Evaluation
The performance of the model was evaluated on DSB2018 data set. The table shows average precision for several IoU thresholds when probability threshold was 0.4, calculated by metric.py script.

IoU threshold	Keras	Pytorch
0.5	0.873	0.8698
0.55	0.85	0.844
0.6	0.8203	0.8078
0.65	0.7612	0.7558
0.7	0.6951	0.7128
0.75	0.5980	0.6336
0.8	0.4770	0.5100
0.85	0.3364	0.3713
0.9	0.1880	0.1932
Notes
