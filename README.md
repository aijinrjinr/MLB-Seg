# MLB-Seg

### Data 
all data will be stored in a folder ```data``` 
```
 MLB-Seg (our repo)
    ├── data├── LA├──train
            |     ├──meta_train
            |     └──original_data
            |
            └── Prostate├──train
                        ├──meta_train
                        └──original_data
```
* Please download the original LA/PROMISE12 dataset and put it in the corresponding ```original_data``` folder which would be used during validation.
* Store each 2D slice from the training set in the corresponding ```train``` folder and each 2D slice from the meta set in the corresponding ```meta_train``` folder.
* For data in ```train```, make sure it has the format shown below
```
XX.npy
  ├──'img'
  ├──'label'
  └──'noisy_label'
```
* 'img' represents the original slice, 'label' is the ground-truth segmentation and 'noisy_label' is the imperfect label (could be noisy annotations or generated labels for unlabelde data)
* For data in ```meta_train```, make sure it has the format shown below
```
XX.npy
  ├──'img'
  └──'label'
```
