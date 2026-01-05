strategy = dict(type='DDPStrategy')
model = dict(type='ConstellationModelRegistry.TimeModel')

iters = 50_000
warmup_iters = 5_000
trainer = dict(
    type='IterBasedTrainer',
    model=model,
    strategy=strategy,
    callbacks=[
        dict(type='OptimizeCallback'),
        dict(
            type='LRScheduleCallback',
            lr_scheduler=dict(
                type='SequentialLR',
                schedulers=[
                    dict(
                        type='LinearLR',
                        start_factor=1e-8,
                        total_iters=warmup_iters - 1,
                    ),
                    dict(
                        type='CosineAnnealingLR',
                        T_max=iters - warmup_iters,
                        eta_min=5e-6,
                    ),
                ],
                milestones=[warmup_iters],
            ),
        ),
        dict(
            type='LogCallback',
            interval=50,
            collect_env=dict(),
            with_file_handler=True,
            eta=dict(type='EMA_ETA', ema=dict(decay=0.9)),
            priority=dict(init=-1),
        ),
        dict(type='GitCallback', diff='HEAD'),
        dict(
            type='TensorBoardCallback',
            interval=50,
            summary_writer=dict(),
            main_tag='train',
        ),
        dict(type='CheckpointCallback', interval=5e3),
    ],
    dataset=dict(
        type='ConstellationDatasetRegistry.TimeDataset',
        split='train',
        batch_size=4096,
    ),
    dataloader=dict(
        type='PrefetchDataLoader',
        batch_size=4,
        num_workers=16,
        sampler=dict(type='DistributedSampler', shuffle=True),
        collate_fn=dict(type='time_collate_fn'),
    ),
    # optimizer=dict(
    #     type='AdamW',
    #     lr=1e-4,
    #     betas=(0.9, 0.98),
    #     weight_decay=1e-4,
    #     eps=1e-8,
    # ),
    optimizer=dict(
        type='SGD',
        lr=1e-1,
        momentum=0.9,
        weight_decay=1e-4,
    ),
    iters=iters,
)
validator = dict(
    type='Validator',
    model=model,
    strategy=strategy,
    callbacks=[
        dict(
            type='MetricCallback',
            metrics=dict(
                tpr0=dict(
                    type='TimeTPRMetric',
                    threshold=0.3,
                    logits='["pred_masks"]',
                    target='["gt_masks"]',
                ),
                fpr0=dict(
                    type='TimeFPRMetric',
                    threshold=0.3,
                    logits='["pred_masks"]',
                    target='["gt_masks"]',
                ),
                precision0=dict(
                    type='TimePrecisionMetric',
                    threshold=0.3,
                    logits='["pred_masks"]',
                    target='["gt_masks"]',
                ),
                recall0=dict(
                    type='TimeRecallMetric',
                    threshold=0.3,
                    logits='["pred_masks"]',
                    target='["gt_masks"]',
                ),
                tpr1=dict(
                    type='TimeTPRMetric',
                    threshold=0.5,
                    logits='["pred_masks"]',
                    target='["gt_masks"]',
                ),
                fpr1=dict(
                    type='TimeFPRMetric',
                    threshold=0.5,
                    logits='["pred_masks"]',
                    target='["gt_masks"]',
                ),
                precision1=dict(
                    type='TimePrecisionMetric',
                    threshold=0.5,
                    logits='["pred_masks"]',
                    target='["gt_masks"]',
                ),
                recall1=dict(
                    type='TimeRecallMetric',
                    threshold=0.5,
                    logits='["pred_masks"]',
                    target='["gt_masks"]',
                ),
                tpr2=dict(
                    type='TimeTPRMetric',
                    threshold=0.7,
                    logits='["pred_masks"]',
                    target='["gt_masks"]',
                ),
                fpr2=dict(
                    type='TimeFPRMetric',
                    threshold=0.7,
                    logits='["pred_masks"]',
                    target='["gt_masks"]',
                ),
                precision2=dict(
                    type='TimePrecisionMetric',
                    threshold=0.7,
                    logits='["pred_masks"]',
                    target='["gt_masks"]',
                ),
                recall2=dict(
                    type='TimeRecallMetric',
                    threshold=0.7,
                    logits='["pred_masks"]',
                    target='["gt_masks"]',
                ),
            ),
        ),
        dict(
            type='LogCallback',
            interval=50,
            collect_env=dict(),
            with_file_handler=True,
            eta=dict(type='EMA_ETA', ema=dict(decay=0.9)),
        ),
    ],
    dataset=dict(
        type='ConstellationDatasetRegistry.TimeDataset',
        split='val_seen',
        batch_size=256,
    ),
    dataloader=dict(
        type='PrefetchDataLoader',
        batch_size=4,
        num_workers=16,
        sampler=dict(type='DistributedSampler', shuffle=False),
        collate_fn=dict(type='time_collate_fn'),
    ),
)
