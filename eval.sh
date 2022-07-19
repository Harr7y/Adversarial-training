#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python eval_cifar10.py --trial 2 --model preresnet
CUDA_VISIBLE_DEVICES=2 python eval_cifar10.py --trial 3 --model wrn