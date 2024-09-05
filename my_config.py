# model settings
_base_ = ["configs/_base_/models/swin/swin_small.py", "configs/default_runtime.py"]
'''

model = dict(  # Config of the model
    type='BMN',  # Type of the localizer
    temporal_dim=100,  # Total frames selected for each video
    boundary_ratio=0.5,  # Ratio for determining video boundaries
    num_samples=32,  # Number of samples for each proposal
    num_samples_per_bin=3,  # Number of bin samples for each sample
    feat_dim=400,  # Dimension of feature
    soft_nms_alpha=0.4,  # Soft NMS alpha
    soft_nms_low_threshold=0.5,  # Soft NMS low threshold
    soft_nms_high_threshold=0.9,  # Soft NMS high threshold
    post_process_top_k=100)  # Top k proposals in post process
# model training and testing settings
train_cfg = None  # Config of training hyperparameters for BMN
test_cfg = dict(average_clips='score')  # Config for testing hyperparameters for BMN


# dataset settings
dataset_type = 'RawframeTrafficDataset'  # Type of dataset for training, validation and testing
data_root = 'data/activitynet_feature_cuhk/csv_mean_100/'  # Root path to data for training
data_root_val = 'data/activitynet_feature_cuhk/csv_mean_100/'  # Root path to data for validation and testing
ann_file_train = 'data/ActivityNet/anet_anno_train.json'  # Path to the annotation file for training
ann_file_val = 'data/ActivityNet/anet_anno_val.json'  # Path to the annotation file for validation
ann_file_test = 'data/ActivityNet/anet_anno_test.json'  # Path to the annotation file for testing


train_pipeline = [  # List of training pipeline steps
    dict(type='LoadLocalizationFeature'),  # Load localization feature pipeline
    dict(type='GenerateLocalizationLabels'),  # Generate localization labels pipeline
    dict(  # Config of Collect
        type='Collect',  # Collect pipeline that decides which keys in the data should be passed to the localizer
        keys=['raw_feature', 'gt_bbox'],  # Keys of input
        meta_name='video_meta',  # Meta name
        meta_keys=['video_name']),  # Meta keys of input
    dict(  # Config of ToTensor
        type='ToTensor',  # Convert other types to tensor type pipeline
        keys=['raw_feature']),  # Keys to be converted from image to tensor
    dict(  # Config of ToDataContainer
        type='ToDataContainer',  # Pipeline to convert the data to DataContainer
        fields=[dict(key='gt_bbox', stack=False, cpu_only=True)])  # Required fields to be converted with keys and attributes
]
val_pipeline = [  # List of validation pipeline steps
    dict(type='LoadLocalizationFeature'),  # Load localization feature pipeline
    dict(type='GenerateLocalizationLabels'),  # Generate localization labels pipeline
    dict(  # Config of Collect
        type='Collect',  # Collect pipeline that decides which keys in the data should be passed to the localizer
        keys=['raw_feature', 'gt_bbox'],  # Keys of input
        meta_name='video_meta',  # Meta name
        meta_keys=[
            'video_name', 'duration_second', 'duration_frame', 'annotations',
            'feature_frame'
        ]),  # Meta keys of input
    dict(  # Config of ToTensor
        type='ToTensor',  # Convert other types to tensor type pipeline
        keys=['raw_feature']),  # Keys to be converted from image to tensor
    dict(  # Config of ToDataContainer
        type='ToDataContainer',  # Pipeline to convert the data to DataContainer
        fields=[dict(key='gt_bbox', stack=False, cpu_only=True)])  # Required fields to be converted with keys and attributes
]
test_pipeline = [  # List of testing pipeline steps
    dict(type='LoadLocalizationFeature'),  # Load localization feature pipeline
    dict(  # Config of Collect
        type='Collect',  # Collect pipeline that decides which keys in the data should be passed to the localizer
        keys=['raw_feature'],  # Keys of input
        meta_name='video_meta',  # Meta name
        meta_keys=[
            'video_name', 'duration_second', 'duration_frame', 'annotations',
            'feature_frame'
        ]),  # Meta keys of input
    dict(  # Config of ToTensor
        type='ToTensor',  # Convert other types to tensor type pipeline
        keys=['raw_feature']),  # Keys to be converted from image to tensor
]
data = dict(  # Config of data
    videos_per_gpu=8,  # Batch size of each single GPU
    workers_per_gpu=8,  # Workers to pre-fetch data for each single GPU
    train_dataloader=dict(  # Additional config of train dataloader
        drop_last=True),  # Whether to drop out the last batch of data in training
    val_dataloader=dict(  # Additional config of validation dataloader
        videos_per_gpu=1),  # Batch size of each single GPU during evaluation
    test_dataloader=dict(  # Additional config of test dataloader
        videos_per_gpu=2),  # Batch size of each single GPU during testing
    test=dict(  # Testing dataset config
        type=dataset_type,
        ann_file=ann_file_test,
        pipeline=test_pipeline,
        data_prefix=data_root_val),
    val=dict(  # Validation dataset config
        type=dataset_type,
        ann_file=ann_file_val,
        pipeline=val_pipeline,
        data_prefix=data_root_val),
    train=dict(  # Training dataset config
        type=dataset_type,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        data_prefix=data_root))

# optimizer
optimizer = dict(
    # Config used to build optimizer, support (1). All the optimizers in PyTorch
    # whose arguments are also the same as those in PyTorch. (2). Custom optimizers
    # which are built on `constructor`, referring to "tutorials/5_new_modules.md"
    # for implementation.
    type='Adam',  # Type of optimizer, refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/optimizer/default_constructor.py#L13 for more details
    lr=0.001,  # Learning rate, see detail usages of the parameters in the documentation of PyTorch
    weight_decay=0.0001)  # Weight decay of Adam
optimizer_config = dict(  # Config used to build the optimizer hook
    grad_clip=None)  # Most of the methods do not use gradient clip
# learning policy
lr_config = dict(  # Learning rate scheduler config used to register LrUpdater hook
    policy='step',  # Policy of scheduler, also support CosineAnnealing, Cyclic, etc. Refer to details of supported LrUpdater from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9
    step=7)  # Steps to decay the learning rate

total_epochs = 9  # Total epochs to train the model
checkpoint_config = dict(  # Config to set the checkpoint hook, Refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py for implementation
    interval=1)  # Interval to save checkpoint
evaluation = dict(  # Config of evaluation during training
    interval=1,  # Interval to perform evaluation
    metrics=['AR@AN'])  # Metrics to be performed
log_config = dict(  # Config to register logger hook
    interval=50,  # Interval to print the log
    hooks=[  # Hooks to be implemented during training
        dict(type='TextLoggerHook'),  # The logger used to record the training process
        # dict(type='TensorboardLoggerHook'),  # The Tensorboard logger is also supported
    ])

# runtime settings
dist_params = dict(backend='nccl')  # Parameters to setup distributed training, the port can also be set
log_level = 'INFO'  # The level of logging
work_dir = './work_dirs/bmn_400x100_2x8_9e_activitynet_feature/'  # Directory to save the model checkpoints and logs for the current experiments
load_from = None  # load models as a pre-trained model from a given path. This will not resume training
resume_from = None  # Resume checkpoints from a given path, the training will be resumed from the epoch when the checkpoint's is saved
workflow = [('train', 1)]  # Workflow for runner. [('train', 1)] means there is only one workflow and the workflow named 'train' is executed once
output_config = dict(  # Config of localization ouput
    out=f'{work_dir}/results.json',  # Path to output file
    output_format='json')  # File format of output file
'''

model = dict(
    backbone=dict(patch_size=(2, 4, 4), drop_path_rate=0.1),
    test_cfg=dict(max_testing_views=4),
)

# dataset settings
dataset_type = "MyDataset"         # 구현한 dataloader 이름
data_root = "/datasets/new_s3_datasets"         # 데이터셋 경로
data_root_val = "/datasets/new_s3_datasets"     # 데이터셋 경로
ann_file_train = "annotation/train_list.txt"    # train annotation 파일 경로
ann_file_val = "annotation/valid_list.txt"      # valid annotation 파일 경로
ann_file_test = "annotation/test_list.txt"      # test annotation 파일 경로
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False
)
train_pipeline = [
    dict(type="SampleFrames", clip_len=1, frame_interval=1, num_clips=1),
    dict(type="RawFrameDecode"),
    dict(type="Resize", scale=(-1, 256)),
    dict(type="RandomResizedCrop"),
    dict(type="Resize", scale=(224, 224), keep_ratio=False),
    dict(type="Flip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="Collect", keys=["imgs", "label"], meta_keys=[]),
    dict(type="ToTensor", keys=["imgs", "label"]),
]
val_pipeline = [
    dict(
        type="SampleFrames", clip_len=1, frame_interval=1, num_clips=1, test_mode=True
    ),
    dict(type="RawFrameDecode"),
    dict(type="Resize", scale=(-1, 256)),
    dict(type="CenterCrop", crop_size=224),
    dict(type="Flip", flip_ratio=0),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="Collect", keys=["imgs", "label"], meta_keys=[]),
    dict(type="ToTensor", keys=["imgs"]),
]
test_pipeline = [
    dict(
        type="SampleFrames", clip_len=1, frame_interval=1, num_clips=1, test_mode=True
    ),
    dict(type="RawFrameDecode"),
    dict(type="Resize", scale=(-1, 224)),
    dict(type="ThreeCrop", crop_size=224),
    dict(type="Flip", flip_ratio=0),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="Collect", keys=["imgs", "label"], meta_keys=[]),
    dict(type="ToTensor", keys=["imgs"]),
]

data = dict(
    videos_per_gpu=64,
    workers_per_gpu=16,
    val_dataloader=dict(videos_per_gpu=1, workers_per_gpu=1),
    test_dataloader=dict(videos_per_gpu=1, workers_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline,
    ),
)
evaluation = dict(interval=5, metrics=["top_k_accuracy", "mean_class_accuracy"])

# optimizer
optimizer = dict(
    type="AdamW",
    lr=1e-3,
    betas=(0.9, 0.999),
    weight_decay=0.02,
    paramwise_cfg=dict(
        custom_keys={
            "absolute_pos_embed": dict(decay_mult=0.0),
            "relative_position_bias_table": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
            "backbone": dict(lr_mult=0.1),
        }
    ),
)
# learning policy
lr_config = dict(
    policy="CosineAnnealing",
    min_lr=0,
    warmup="linear",
    warmup_by_epoch=True,
    warmup_iters=2.5,
)
total_epochs = 10

# runtime settings
checkpoint_config = dict(interval=1)
work_dir = "./work_dirs/small_pretrainedX_cls14.py"
find_unused_parameters = False
log_level = "INFO"

# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=8,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)
