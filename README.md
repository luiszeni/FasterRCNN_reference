
1- clone the repo
```
git clone https://github.com/luiszeni/FasterRCNN_reference.git && cd FasterRCNN_reference
```

2-create the dir to extract the dataset
```
mkdir data
mkdir data/coco
cd data/coco 
```

3-download the coco data from the site
```
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```

4-extract coco data
```
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip
```


It will have the following structure
  ```
  coco
  ├── annotations
  │   ├── instances_train2017.json
  │   ├── instances_val2017.json
  ├── train2017
  ├── val2017
  ```

5-return to root directory
```
cd ../..
```


6-build the docker machine
```
cd docker
docker build . -t faster
cd ..
```

7-create a container
```
docker run --gpus all -v  $(pwd):/root/faster_reference --shm-size 12G -ti --name faster faster
```

8-put the model to train


Single gpu training:
```
python3  code/tasks/train.py --dataset coco --model fasterrcnn_resnet50_fpn --epochs 26 --batch-size 4 --lr 0.005
```

Multi gpu-trianing (I am not sure if it work):
```
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env code/tasks/train.py\
    --dataset coco --model fasterrcnn_resnet50_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3
```

9-Evaluate on val set:
```
TODO
```
