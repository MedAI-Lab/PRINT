import math
from encoder_schedule import depth, encoder_lr
from spatial_residual_training import (
    epoch_audit as spatial_profile_audit,
    record_gradient as record_spatial_gradient, spatial_objective,
    map_correlation_or_none,
    singleton_pass,
    stage_technical_audit, technical_map_audit,
)
def check_encoder_schedule(model, core, e, d, lr, cap):
    if d != depth(e) or (not math.isclose(lr, encoder_lr(e), abs_tol=1e-15)) or (cap != 0.3):
        raise RuntimeError('Training parameters differ from the specified epoch schedule')
    for (name, p) in model.named_parameters():
        if not core.is_encoder_name(name):
            continue
        expected = name.startswith('norm.') and d > 0 or (name.startswith('blocks.') and int(name.split('.')[1]) >= 12 - d)
        if p.requires_grad != expected:
            raise RuntimeError('Unexpected trainability for encoder parameter ' + name)
