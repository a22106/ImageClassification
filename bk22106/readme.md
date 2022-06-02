1. 학습 준비
1.1 COCO 데이터 세트를 이용한 학습
크기가 커서 curl을 이용하여 다운받는 것을 추천
COCO 데이터 세트 종류
이미지

2014 Train images [83K/13GB]
2014 Val images [41K/6GB]
2014 Test images [41K/6GB]
2015 Test images [81K/12GB]
2017 Train images [118K/18GB]
2017 Val images [5K/1GB]
2017 Test images [41K/6GB]
2017 Unlabeled images [123K/19GB]


주석

2014 Train/Val annotations [241MB]
2014 Testing Image info [1MB]
2015 Testing Image info [2MB]
2017 Train/Val annotations [241MB]
2017 Stuff Train/Val annotations [1.1GB]
2017 Panoptic Train/Val annotations [821MB]
2017 Testing Image info [1MB]
2017 Unlabeled Image info [4MB]

curl 다운로드
$ sudo apt install curl$ curl https://sdk.cloud.google.com | bash$ source ~/.bashrc
 
COCO 데이터 세트 다운로드
$ mkdir COCO$ cd COCO$ mkdir val2017$ gsutil -m rsync gs://images.cocodataset.org/val2017 val2017$ mkdir annotation$ gsutil -m rsync gs://images.cocodataset.org/annotations annotation$ cd val2017$ unzip ../anns/annotations_trainval2017.zip
 
다른 데이터는 다음과 같이 받으면 된다.
$ gsutil -m rsync gs://images.cocodataset.org/train2017 train2017
$ !wget http://images.cocodataset.org/zips/train2017.zip

val2017 이미지 데이터는 5,000장으로 이류어져있으며, annotation은 다음과 같이 구성됨
* caption: 영상에 대한 설명
* person_keypoints: 사람의 관절 좌표 데이터
* instances: 영상에 포함된 사람 혹은 사물에 대한 카테고리와 Segmentation 정보

COCO데이터 세트는 COCO API를 이용하여 다룰 수 있다.
COCO API: https://github.com/cocodataset/cocoapi

COCO 데이터 주석 확인은 아래와 같이 json viewer를 통해 볼 수 있다.
JSON Viewer: http://jsonviewer.stack.hu/

1.2 데이터 포맷을 Yolo 형식으로 바꾸기
COCO데이터 포맷은 bbox 값이 x, y, w, h 값으로 구성되어있음
YOLO에서의 포맷은 클래스 번호와 전체 영상 크기에 대한 center x, center y, w, h 비율 값으로 구성된다.

또한 COCO는 모든 영상의 주석이 담겨진 하나의 json 파일로 구성되어있으며, YOLO는 한 영상당 한 개의 txt파일로 구성되어있음

1.5 Dartnet 다운로드
데이터 준비가 끝나면 Darknet을 다운로드함
$ git clone https://github.com/pjreddie/darknet.git

* COCOTOYOLO-Annotation
java -jar cocotoyolo.jar ./instances_train2017.json ./train2017 "person, bicycle, car, motorbike, aeroplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, sofa, pottedplant, bed, diningtable, toilet, tvmonitor, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush" ./success