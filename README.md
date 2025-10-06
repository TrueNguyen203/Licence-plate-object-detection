# Lisence plate object detection

## Description
This is one of my project in university. The goal of this project is to make a regression deep learning model that can predict the bounding box of lisence plates on cars.

## Data description
The data a public dataset on hugging face which you can find in this link 
https://huggingface.co/datasets/keremberke/license-plate-object-detection

The data have more than 8000 which have already been augumentated.
Data format: </br>
`{'image_id', 'image', 'width', 'height', objects : { 'id', 'area', 'bbox', 'category'} }`

--image--

## My Aproach
* For this object detection problem, I am using `fasterrcnn_restnet_50_fpn` model with the pretrained weights on the COCO dataset.
* The model data architecture is the combination of the `resnet-50` (redidual netwrok) backbone and the `FPN` (Feature pyramid network) head
* I have recustomized the aspect ratio for lisence plate since it has relatively bigger width compare to the height: </br>
    `(anchor_sizes = ((32,), (64,), (128,), (256,), (512,)))`</br>
    `aspect_ratios = ((0.25, 0.5, 1.0, 2.0, 4.0),) * len(anchor_sizes)`
* Loss of the model will be the total of `{'loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg'} `

## Evaluation of the model:
* Loss dict & Average loss perbatch of 4:
--image--
* mAp (Mean Average Precision) Score with IoU threshold 0.5 and 0.75: </br>
  `'map_50': tensor(0.9798), 'map_75': tensor(0.8414)`
* Example images of the correct prediction:
  --image--
* Example images of the wrong prediction:
-- image --

## Limitations:

  
