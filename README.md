

1- clone the repo 
```
git clone https://github.com/luiszeni/FasterRCNN_reference.git && cd FasterRCNN_reference
```

2-create the dir to extract the dataset
```
mkdir data
cd data 
```

3- Download the training, validation, test data, and VOCdevkit
    ```Shell
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
    ```
    Optional, normally faster to download, links to VOC (from darknet):
    ```Shell
    wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
    wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
    ```
4- Extract all of these tars into one directory named `VOCdevkit`
    ```Shell
    tar xvf VOCtrainval_06-Nov-2007.tar
    tar xvf VOCtest_06-Nov-2007.tar
    ```
5- Download pascal annotations in the COCO format
    ```Shell
    wget http://inf.ufrgs.br/~lfazeni/CVPR_deepvision2020/coco_annotations_VOC.tar
    ```
6- Extract the annotations
    ```Shell
    tar xvf coco_annotations_VOC.tar
    ```

It should have this basic structure
    ```Shell
    $VOC2007/                           
    $VOC2007/annotations
    $VOC2007/JPEGImages
    $VOC2007/VOCdevkit        
    # ... and several other directories ...
    ```

7-return to root directory
```
cd ../..
```


8-build the docker machine
```
cd docker
docker build . -t faster
cd ..
```

9-create a container
```
docker run --gpus all -v  $(pwd):/root/faster_reference --shm-size 12G -ti --name faster faster
```

10-put the model to train


Single gpu training:
```
python3  code/tasks/train.py --dataset voc --data-path data/VOCdevkit/VOC2007/ --model fasterrcnn_resnet50_fpn --epochs 64 --batch-size 6
```

11-Evaluate on test set:
```
python3  code/tasks/train.py --dataset voc --data-path data/VOCdevkit/VOC2007/ --model fasterrcnn_resnet50_fpn --test_only --resume <location of the weights>
```



## giou was trained using:
python3  code/tasks/train.py --dataset voc --data-path data/VOCdevkit/VOC2007/ --model fasterrcnn_resnet50_fpn --epochs 64 --batch-size 6 --loss-bbox-type giou --loss-rpn-type giou --loss-bbox-weight 10