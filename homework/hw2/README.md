# HW2 
Computed Tomography Lung Tumor Segmentation Task

## 1. Data Preprocess

* Clean up the training data: remove 12 patients with wrong label
* Transfer to Housefield Unit (HU)
* Normalize to 0 ~ 1 with lower threshold = -1000, upper threshold = 400
 
## 2. Save preprocessed data into npz files

## 3. Train 2D U-net model

* validation size = 0.1
* Loss: binary_crossentropy
* Optimizer: Adam 
* Learning rate: 1e-3
* Batch size: 8
* Epoch: 25

## 4. Predict with model epoch 20

* Private/Public leaderboard: 0.41851 / 0.35392
