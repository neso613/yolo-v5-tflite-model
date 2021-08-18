# YOLO-v5 TFLite Model

YOLOv5 - most advanced vision AI model for object detection. Natively implemented in PyTorch and exportable to TFLite for use in edge solutions. This repository provides an Object Detection model in TensorFlow Lite (TFLite) for TensorFlow 2.x. These models primarily come from two repositories - [ultralytics](https://github.com/ultralytics/yolov5) and [zldrobit](https://github.com/zldrobit/yolov5). We provide end-to-end code that show the inference process using TFLite and model conversion.\
[English-ASR pip wheel](https://pypi.org/project/english-asr/1.2/)\
[TFHub](https://tfhub.dev/neso613/lite-model/yolo-v5-tflite/tflite_model/1)

## Installation
- pip3 install -r requirements.txt

## Pretrained Checkpoints

|Model |size<br><sup>(pixels) |mAP<sup>val<br>0.5:0.95 |mAP<sup>test<br>0.5:0.95 |mAP<sup>val<br>0.5 |Speed<br><sup>V100 (ms) | |params<br><sup>(M) |FLOPs<br><sup>640 (B)
|---            |---  |---      |---      |---      |---     |---|---   |---
|[YOLOv5s]      |640  |36.7     |36.7     |55.4     |**2.0** |   |7.3   |17.0
|[YOLOv5m]      |640  |44.5     |44.5     |63.1     |2.7     |   |21.4  |51.3
|[YOLOv5l]      |640  |48.2     |48.2     |66.9     |3.8     |   |47.0  |115.4
|[YOLOv5x]      |640  |**50.4** |**50.4** |**68.8** |6.1     |   |87.7  |218.8
|                       |     |         |         |         |        |   |      |
|[YOLOv5s6]     |1280 |43.3     |43.3     |61.9     |**4.3** |   |12.7  |17.4
|[YOLOv5m6]     |1280 |50.5     |50.5     |68.7     |8.4     |   |35.9  |52.4
|[YOLOv5l6]     |1280 |53.4     |53.4     |71.1     |12.3    |   |77.2  |117.7
|[YOLOv5x6]     |1280 |**54.4** |**54.4** |**72.0** |22.4    |   |141.8 |222.9
|                       |     |         |         |         |        |   |      |
|[YOLOv5x6]     TTA |1280 |**55.0** |**55.0** |**72.0** |70.8    |   |-     |-


  <summary>Table Notes</summary>
  
  * AP<sup>test</sup> denotes COCO [test-dev2017](http://cocodataset.org/#upload) server results, all other AP results denote val2017 accuracy.  
  * AP values are for single-model single-scale unless otherwise noted. **Reproduce mAP** by `python test.py --data coco.yaml --img 640 --conf 0.001 --iou 0.65`  
  * Speed<sub>GPU</sub> averaged over 5000 COCO val2017 images using a GCP [n1-standard-16](https://cloud.google.com/compute/docs/machine-types#n1_standard_machine_types) V100 instance, and includes FP16 inference, postprocessing and NMS. **Reproduce speed** by `python test.py --data coco.yaml --img 640 --conf 0.25 --iou 0.45`  
  * All checkpoints are trained to 300 epochs with default settings and hyperparameters (no autoaugmentation). 
  * Test Time Augmentation ([TTA](https://github.com/ultralytics/yolov5/issues/303)) includes reflection and scale augmentation. **Reproduce TTA** by `python test.py --data coco.yaml --img 1536 --iou 0.7 --augment`


## References
- [TensorFlow Lite Conversion](https://www.tensorflow.org/lite/convert)
- [Float16 quantization in TensorFlow Lite](https://www.tensorflow.org/lite/performance/post_training_float16_quant)
- [Dynamic-range quantization in TensorFlow Lite](https://www.tensorflow.org/lite/performance/post_training_quant)
