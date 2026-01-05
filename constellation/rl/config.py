environment = dict(world_size=0)

algorithm = dict(
    n_steps=20,
    batch_size=4,
    verbose=1,
)

learn = dict(total_timesteps=100000,
             # progress_bar=True,
             )

callbacks = [
    dict(
        type='CheckpointCallback',
        save_freq=10_000 + 1,
        save_path='./work_dirs/',
        name_prefix='ppo_ckpt',
        verbose=2,
    ),
]
