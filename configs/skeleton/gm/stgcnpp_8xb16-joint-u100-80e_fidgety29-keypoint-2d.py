_base_ = '../../_base_/default_runtime.py'

# load_from = 'https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu120-xsub-keypoint-2d/stgcn_8xb16-joint-u100-80e_ntu120-xsub-keypoint-2d_20221129-612416c6.pth'

model = dict(
    type='RecognizerWSTAL',
    backbone=dict(
        type='STGCN',
        gcn_adaptive='init',
        gcn_with_res=True,
        tcn_type='mstcn',
        graph_cfg=dict(layout='gm', mode='spatial'),
        in_channels=3),
    cls_head=dict(type='GCNHead', num_classes=2, in_channels=256),
    weight=[651/595/2, 651/56/2, 1],
    # weight=[1, 1],
    topk=(1/4, 1/6, 1/6),
    freeze=False)

dataset_type = 'PoseDataset'
ann_file = 'data/sorted_GMA29.pkl'
train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='AlignPose', dataset='gm', aug=True, rand_angle=(-30, 30), rand_scale=(2/3, 4/3), rand_trans=(-1/3, 1/3)),
    dict(type='GenSkeFeat', dataset='gm', feats=['j']),
    dict(type='SampleAllFrames', clip_len=150, num_clips=23, clip_interval=75, frame_interval=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='AlignPose', dataset='gm'),
    dict(type='GenSkeFeat', dataset='gm', feats=['j']),
    dict(
        type='SampleAllFrames', clip_len=150, clip_interval=75, frame_interval=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='AlignPose', dataset='gm'),
    dict(type='GenSkeFeat', dataset='gm', feats=['j']),
    dict(
        type='SampleAllFrames', clip_len=150, clip_interval=75, frame_interval=1,
        test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=8,
    num_workers=30,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            ann_file=ann_file,
            pipeline=train_pipeline,
            split='writhing_train')))
val_dataloader = dict(
    batch_size=1,
    num_workers=30,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=val_pipeline,
        split='writhing_test',
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=30,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=test_pipeline,
        split='writhing_test',
        test_mode=True))

# val_evaluator = [dict(type='SimpleLoss')]
val_evaluator = [dict(type='AccMetric', metric_list=('mean_average_precision',))]
test_evaluator = val_evaluator

param_scheduler = [
    dict(
        type='ExponentialLR',
        gamma = 0.8)
]

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=20, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01))

default_hooks = dict(checkpoint=dict(interval=1), logger=dict(interval=100))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (16 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=128)
