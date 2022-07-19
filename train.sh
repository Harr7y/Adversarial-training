#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python at.py --trial 13 --model wrn --fre_loss True --epochs 200 --lr_drop 75,150
