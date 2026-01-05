trainer = dict(
    type='BaseTrainer',
    dataset=dict(
        type='SatDatasetRegistry.SatDataset',
        access_layer=dict(
            type='SatAccessLayerRegistry.JsonAccessLayer',
            data_root='syn_data',
        ),
    ),
    dataloader=dict(
        batch_size=1,
        num_workers=0,
        shuffle=True,
    ),
    model=dict(type='SatModelRegistry.VanillaModel'),
    optimizer=dict(
        type='AdamW',
        lr=1e-4,
        betas=(0.9, 0.98),
        weight_decay=0.2,
        eps=1e-6,
    ),
    callbacks=[
        dict(
            type='LogCallback',
            interval=1,
            collect_env=dict(),
            with_file_handler=True,
            eta=dict(type='EMA_ETA', ema=dict(decay=0.9)),
            priority=dict(init=-1),
        ),
    ],
    strategy=dict(type='DDPStrategy'),
    iters=10,
)
