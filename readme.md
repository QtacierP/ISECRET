# I-SECRET

This is the implementation of the MICCAI 2021 Paper "I-SECRET: Importance-guided fundus image enhancement via semi-supervised contrastive constraining". 

## Data preparation
1. Firstly, download EyeQ dataset from [EyeQ](https://github.com/HzFu/EyeQ).
2. Split the dataset into train/val/test according to the [EyePACS challenge](https://www.kaggle.com/c/diabetic-retinopathy-detection).
3. Run
```bash
python tools/degrade_eyeq.py --degrade_dir ${DATA_PATH}$ --output_dir $OUTPUT_PATH$ --mask_dir ${MASK_PATH}$ --gt_dir ${GT_PATH}$.
```
Note that this scipt should be applied for usable dataset for cropping pre-processing.

4. Make the architecture of the EyeQ directory as:
```bash
.
├── 
├── train
│   └── crop_good
│   └── degrade_good
│   └── crop_usable
├── val
│   └── crop_good
│   └── degrade_good
│   └── crop_usable
├── test
│   └── crop_good
│   └── degrade_good
│   └── crop_usable
```
Here, the crop_good is the \${GT_PATH}\$ in the step 3, and degrade_good is the \${OUTPUT_PATH}\$ in the step 3.

## Package install
Run
``` bash
pip install -r requirements.txt
```


## Run pipeline
Run the baseline model
```bash
python main.py --model i-secret --lambda_rec 1 --lambda_gan 1 --data_root_dir ${DATA_DIR}$ --gpu ${GPU_INDEXS}$ -- batch size {BATCH_SIZE}$  --name baseline --experiment_root_dir ${LOG_DIR}$
```

Run the model with IS-loss
```bash
python main.py --model i-secret --lambda_is 1 --lambda_gan 1 --data_root_dir ${DATA_DIR}$ --gpu ${GPU_INDEXS}$ -- batch size {BATCH_SIZE}$  --name is_loss --experiment_root_dir ${LOG_DIR}$
```

Run the I-SECRET model 
```bash
python main.py --model i-secret --lambda_is 1 --lambda_icc 1 --lambda_gan 1 --data_root_dir ${DATA_DIR}$ --gpu ${GPU_INDEXS}$ --batch_size {BATCH_SIZE}$  --name i-secret --experiment_root_dir ${LOG_DIR}$
```
## Visualization
Go to the \${LOG_DIR}\$ / \${EXPERIMENT_NAME}\$ / checkpoint, run
```bash
tensorboard --logdir ./ --port ${PORT}$
```
then go to localhost:\${PORT}\$ for detailed logging and visualization.

## Test and evalutation
Run 
```bash
python main.py --test --resume 0 --test_dir ${INPUT_PATH}$ --output_dir ${OUTPUT_PATH}$ --name ${EXPERIMENT_NAME}$ --gpu ${GPU_INDEXS}$ -- batch size {BATCH_SIZE}$ 
```
Please note that the metric outputted by test script is under the PyTorch pre-process (resize etc.). It is not precise. Therefore, we need to run the evaluation scipt for further evaluation.
``` bash
python tools/evaluate.py --test_dir ${OUTPUT_PATH}$ --gt_dir ${GT_PATH}$
```
## Vessel segmentation
We apply the iter-Net framework. We simply replace the test set with the degraded images/enhanced images. For more details, please follow [IterNet](https://github.com/conscienceli/IterNet). 

## Future Plan
- [ ] Cleaning codes
- [ ] More SOTA backbones (ResNest ...)
- [ ] WGAN loss
- [ ] Internal evaluations for down-sampling tasks

## Acknowledgment
Thanks for [CutGAN](https://github.com/taesungp/contrastive-unpaired-translation) for the implementation of patch NCE loss, [EyeQ_Enhancement](https://github.com/HzFu/EyeQ_Enhancement) for degradation codes, [Slowfast](https://github.com/facebookresearch/SlowFast) for the distributed training codes

