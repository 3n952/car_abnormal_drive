# optimizer
optimizer = dict(
    type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001  # this lr is used for 8 gpus
)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy="step", step=[20, 40])
total_epochs = 50
