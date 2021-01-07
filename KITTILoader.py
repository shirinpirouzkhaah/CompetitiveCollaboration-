# -*- coding: utf-8 -*-
"""

@author: Shirin
"""

from Dataloader import Kittiloader
...

loader = Kittiloader(<kitti_root_path>, <'train', 'val' or 'test'>)
data_size = loader.data_length() # get data split size
data_item = loader.load_item(idx) # which is very suitable for pytorch dataloader

from dataset import DataGenerator
...

# transformer will be defined automatically according to phase once datagen instance is created
datagen = DataGenerator(<kitti_root_path>, phase=<'train', 'val' or 'test'>)
kittidataset = datagen.create_data(batch_size)

# other code before training loop
...

for epoch in range(num_epoches):
    # training loop for an epoch
    for id, batch in enumerate(kittidataset):
        # various types of data can be acquired here
        left_img_batch = batch['left_img'] # batch of left image, id 02
        right_img_batch = batch['right_img'] # batch of right image, id 03
        depth_batch = batch['depth'] # the corresponding depth ground truth of given id
        depth_interp_batch = batch['depth_interp'] # dense depth for visualization
        fb = batch['fb'] # focal_length * baseline
        
