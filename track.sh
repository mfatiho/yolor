#!/bin/bash
for seq in MOT16-02 MOT16-04 MOT16-05 MOT16-09 MOT16-10 MOT16-11 MOT16-13
do
	python detect_deepsort.py --source ../../Datasets/MOT16/train/${seq}/img1/ --cfg cfg/yolor_p6.cfg --weights yolor_p6.pt --conf-thres 0.25 --img-size 1280 --device 0 --save-all-txt ${seq} --class 0
done

python -m motmetrics.apps.eval_motchallenge ../../Datasets/MOT16/train/ inference/dets/

