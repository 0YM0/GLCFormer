custom_imports = dict(imports=[ 
    'mmgeo.datasets',  
    'mmgeo.evaluation',
    'mmgeo.hooks',
    'mmgeo.models',
    'mmgeo.visualization',
    #'mmseg.models',
    'mmdet.models'
    ]
)   
default_scope = 'mmseg'

batch_size = 16
num_classes = 1
num_channels = 3   
num_workers = 8

ignore_index = 255 
  
max_iters = 25000 
warmup_iter = max(1, int(0.1*max_iters)) 
by_epoch = False
val_interval = 500

lr = 1e-4 ###### 
wd = 0.01 ######

train_loop = 'IterBasedTrainLoop' #'EpochBasedTrainLoop', 'IterBasedTrainLoop'
log_interval = 100
draw_img = False # True
draw_interval = 500
train_cfg = dict(max_iters=max_iters, type=train_loop, val_interval=val_interval)

seed = 42
randomness = dict(seed=seed)
resume = False  

log_level = 'INFO'
debug = False #True
 
image_size = (512,512)
test_stride = (384,384)

run_name = '[Dual_Branch_GeoLink_v3]_[Best_model_Swin_B_UperNet]_[Spatial_split]_[80_10_10]_LandCoverGraph_WarmStartSpatialBiasTwoWayFusion_iter25k_chip512_batch16_CE_LR1e4_Binary_train_semseg' 
prj_dir = "./"
data_root = f'{prj_dir}/data/exp3_250818' 
train_graph_dir = 'graphs_hier/train_512_landcover_wkt'
val_graph_dir = 'graphs_hier/validation_slide_512_384_landcover_wkt'
test_graph_dir = 'graphs_hier/test_slide_512_384_landcover_wkt'
work_dir = f'{prj_dir}/work_dirs/dual_branch_Swin_B/Swin_B_UperNet/Graph_model_GeoLink_v3/split_ratios/{run_name}/Test/Base' # /Test/Base (Test 할 때 적용)


vis_backends = [
        dict(type='LocalVisBackend'), 
        #dict(type='TensorboardVisBackend',
        dict(type='WandbVisBackend',
            save_dir= f"{work_dir}/wandb_log", #로그를 저장할 로컬 디렉토리 경로  
            init_kwargs = dict(project = 'Satellite',  name = f'{run_name}',  ),
            watch_kwargs = dict(log='all',  log_freq=log_interval)
        ) 
]

############################################# WandB 연동코드
"""
visualizer = dict(
    name='visualizer',
    type= 'PatchSegLocalVisualizer', #PatchSegLocalVisualizer
    vis_backends= vis_backends
)  
"""

#>>> DATASET >>>    
train_dataset = dict(
    type='KoreabarnListDataset', 
    data_list_path = f"data_list_final/spatial_split/base_model/sampling/80_10_10/train_sample_512_ocs10_rs10.csv", 
    data_root = data_root, 
    img_dir = 'images/recent/', 
    mask_dir = 'masks/recent/binary_inmap_roi',
    seg_map_suffix = '.tif', 
    label_dir = 'labels',
    debug = debug,
    use_png = False,
    graph_dir = train_graph_dir,
    graph_required = False,
    pipeline=[
        dict(type='CustomLoadAnnotations',
            patch_type = 'fixed',
            ignore_index = ignore_index,
            crop_size= image_size, 
            ),    
        dict(type='CustomLoadImageFromFile',
            patch_type = 'fixed',
            ignore_index = ignore_index,
            ),
        dict(type='LoadLandCoverGraph',
            required = False,
            allow_missing = True,
            ),
        ##### Data Augmentation
        dict(type='PhotoMetricDistortion'),
        #####
        dict(type='PackSegInputsWithGraph',
            meta_keys=('img_path', 'seg_map_path', 'ori_shape',
                    'img_shape', 'pad_shape', 'scale_factor', 'flip',
                    'flip_direction', 'reduce_zero_label',
                    'read_bbox', 'rotate_deg', 'crop_size', 'graph_path') #추가 
        )
    ] ,
)
 
train_dataloader = dict(
    batch_size=batch_size,
    dataset= train_dataset, 
    num_workers=num_workers,
    persistent_workers= True,  
    sampler=dict(shuffle=True, type='DefaultSampler'))

#>>>
val_dataset =  dict(
    type='KoreabarnListDataset', 
    data_root = data_root, 
    img_dir = 'images/recent/', 
    mask_dir = 'masks/recent/binary_inmap_roi', 
    seg_map_suffix = '.tif', 
    label_dir = 'labels',
    data_list_path = f"data_list_final/spatial_split/base_model/split/80_10_10/validation_file_list.csv", 
    debug = debug,
    use_png = False,
    graph_dir = val_graph_dir,
    graph_required = False,
    pipeline=[

        dict(type='CustomLoadAnnotations',
            #patch_type = 'fixed' , 
            ignore_index = ignore_index,
            #crop_size= image_size, 
        ),    
        dict(type='CustomLoadImageFromFile',
            #patch_type =  'fixed',  
            ignore_index = ignore_index,
        ), 
        dict(type='PackSegInputs',
            meta_keys=('img_path', 'seg_map_path', 'ori_shape',
                    'img_shape', 'pad_shape', 'scale_factor', 'flip',
                    'flip_direction', 'reduce_zero_label',
                    'read_bbox', 'graph_dir',
                    #'rotate_deg', 'crop_bbox'
                    )   
        ),
    ] ,
)
val_dataloader = dict(
    batch_size=1, #에서 배치를 쌓음 ***** 
    dataset=val_dataset, 
    num_workers=num_workers,
    persistent_workers=False ,
    sampler=dict(shuffle=False, type='DefaultSampler')) 


test_dataset =  dict(
    type='KoreabarnListDataset', 
    data_root = data_root, 
    img_dir = 'images/recent/',  
    mask_dir = 'masks/recent/binary_inmap_roi', 
    seg_map_suffix = '.tif', 
    label_dir = 'labels',
    data_list_path = f"data_list_final/spatial_split/base_model/split/80_10_10/test_file_list.csv",  
    debug = debug,
    use_png = False,
    graph_dir = test_graph_dir,
    graph_required = False,
    pipeline=[ 
        dict(type='CustomLoadAnnotations', 
            ignore_index = ignore_index, 
        ),    
        dict(type='CustomLoadImageFromFile', 
            ignore_index = ignore_index,
            bands=[1, 2, 3], #############
        ), 
        dict(type='PackSegInputs',
            meta_keys=('img_path', 'seg_map_path', 'ori_shape',
                    'img_shape', 'pad_shape', 'scale_factor', 'flip',
                    'flip_direction', 'reduce_zero_label',
                    'read_bbox', 'graph_dir',
                    #'rotate_deg', 'crop_bbox'
                    )   
        ),
    ] ,
)
test_dataloader = dict(
    batch_size=1, 
    dataset=test_dataset, 
    num_workers=2 ,
    persistent_workers=False ,
    sampler=dict(shuffle=False, type='DefaultSampler'))  


model_test_cfg = dict(
    mode = 'slide', 
    crop_size = image_size, 
    stride = test_stride ,
    patch_batch_size = 1, 
    ignore_index = ignore_index, 
) 


norm_cfg = dict(requires_grad=True, type='SyncBN')
data_preprocessor = dict(  #여기서 패딩을 진행 
    pad_val=0,
    seg_pad_val=255,
    size=image_size, 
    mean=None,
    std=None,
    type='CustomSegDataPreProcessor')


################
pretrained = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_base_patch4_window12_384_22k_20220317-e5c09f74.pth'  # noqa

model = dict(
    type='SlideEncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        frozen_stages=-1,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    graph_fusion=dict(
        type='SwinGeoLinkWarmStartLandCoverGraphFusion',
        image_in_channels=1024,
        embed_dims=512,
        image_feature_index=-1,
        num_heads=8,
        num_graph_layers=3,
        graph_num_heads=8,
        graph_ffn_ratio=4.0,
        graph_use_pos=True,
        num_fusion_layers=1,
        ffn_ratio=4.0,
        fusion_residual_mode='delta',
        residual_scale_init=0.05,
        residual_scale_trainable=True,
        return_attention=False,
        keep_scalar_gate=False,
        pos_num_frequencies=16,
        spatial_distance_scale=2.0,
        spatial_bbox_scale=1.0,
        spatial_inside_bias=0.5,
        learnable_spatial_bias=True,
        enable_landcover_aux_loss=False,
        landcover_aux_loss_weight=0.0,
        landcover_aux_min_spatial_score=-2.0,
        dropout=0.1,
        graph_gate_init=0.05),
    decode_head=dict(
        type='UPerHead',
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg = model_test_cfg)

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=lr, type='AdamW', weight_decay=wd),
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            norm=dict(decay_mult=0.0),
            pos_block=dict(decay_mult=0.0))),
    type='OptimWrapper')
param_scheduler = [
    dict(begin=0, by_epoch=by_epoch, end=warmup_iter, start_factor=1e-06, type='LinearLR'),
    dict(
        begin=warmup_iter,
        by_epoch=by_epoch,
        end=max_iters,
        eta_min=1e-05,
        power=0.99,
        type='PolyLR'),
] 

# learning policy
param_scheduler = [
  dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=warmup_iter),
  dict(type='PolyLR', eta_min=0, power=0.9, by_epoch=False, begin=warmup_iter, end=max_iters),
]


val_cfg = dict(type='ValLoop') 
val_evaluator = dict(iou_metrics=[ 'mIoU', 'mFscore', 'mDice'], 
            ignore_index = ignore_index,
            type='PositiveIoUMetric')
#>>>
test_cfg = dict(type='TestLoop') 
test_evaluator = dict(iou_metrics=[ 'mIoU', 'mFscore', 'mDice' ], 
            ignore_index = ignore_index,
            type='PositiveIoUMetric')


# >>> MISC >>>       
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
log_processor = dict(by_epoch=by_epoch)



default_hooks = dict(
    checkpoint=dict(
        by_epoch=by_epoch, 
        max_keep_ckpts=-1,
        #save_begin=10,
        #interval=10,
        save_best='mIoU', #mIoU
        type='CheckpointHook'),
    logger=dict(interval=log_interval, 
                log_metric_by_epoch=by_epoch, 
                type='LoggerHook',
            ),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    #visualization=dict(draw=draw_img, 
    #        interval = draw_interval, 
    #        type=PatchSegVisualizationHook, #PatchSegVisualizationHook
    #        #'SegVisualizationHook'
    #)
)
