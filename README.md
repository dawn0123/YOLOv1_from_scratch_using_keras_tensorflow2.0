# YOLOv1_from_scratch_using_keras_tensorflow2.0

<p>
    <img src="architecture.png" />
</p>

__The Architecture.__
_Our detection network has 24 convolutional layers followed by 2 fully connected layers. Alternating 1 x 1 convolutional layers reduce the features space from preceding layers. We pretrain the convolutional layers on the ImageNet classification task at half the resolution (224 x 224 input image) and then double the resolution for detection._

This is an implementation of YOLOv1 as described in the paper [__You Only Look Once__](https://arxiv.org/abs/1506.02640).

I used [__VOC 2007__](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/) dataset as its size is manageable so it  would be easy to un it using Google Colab.

First, you should download dataset and set its path in config.py.

You shoud preprocess dataset.

```
python data_preprocess.py
```

When you are ready you can train your customized YOLOv1 model.

```
python train.py
```

## Result

<p>
    <img src="city.jpg" />
    <img src="city_pred.jpg" />
</p>


## Conclusion

This implementation won’t achieve the same accuracy as what was described in the paper since we have skipped the pretraining step.