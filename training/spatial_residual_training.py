"""Spatial residual training and gradient checks."""
import math
import torch
import torch.nn.functional as F
import spatial_training_runtime as parent
from spatial_modules import install_spatial_arm

_ROWS = {}


def post_model(arm, model):
    before = {name:p.detach().cpu().clone() for name,p in model.named_parameters()}
    rng = torch.get_rng_state().clone()
    audit = install_spatial_arm(model, arm)
    model.spatial_initial_state = {k: v.detach().cpu().clone() for k,v in model.factor_v12_spatial.state_dict().items()}
    after = dict(model.named_parameters())
    checks = {
        'singleton_enabled':model.factor_v12_spatial_gate.enable_single_image_gate and model.factor_v12_spatial_gate.single_image_scale==1.,
        'inherited_weights_bitwise_preserved':all(name in after and torch.equal(value,after[name].detach().cpu()) for name,value in before.items()),
        'agate_weights_bitwise_preserved':all(torch.equal(value,after[name].detach().cpu()) for name,value in before.items() if name.startswith('factor_v12_agate')),
        'cpu_rng_bitwise_preserved':torch.equal(rng,torch.get_rng_state()),
        'arm_matches_module':model.factor_v12_spatial.v31_arm==arm,
        'cap_matches_plan':model.factor_v12_spatial.maximum_residual_z==.15,
    }
    if not all(checks.values()):
        raise RuntimeError(f'installation audit failed: {checks}')
    model.v31_arm_audit = dict(audit, status='PASS', branch_mode='residual',
        singleton_spatial_enabled=checks['singleton_enabled'], checks=checks)
    model.spatial_initialization_audit['v31_runtime_model_audit'] = model.v31_arm_audit


def spatial_objective(fidelity, flip_loss, details, target):
    prediction = details['agate_prediction'].detach().float() + details['spatial_correction'].float()
    profile = .10 * (1-F.cosine_similarity(prediction.float(), target.float(), dim=1)).mean()
    if not torch.isfinite(profile):
        raise FloatingPointError('Profile non-finite')
    return fidelity + .10*flip_loss + profile, profile


def record_gradient(arm, epoch, component, gradients, parameters, model):
    parent.record_spatial_gradient(parent.SPATIAL_PROFILE_ARM, epoch, component, gradients, parameters, model)
    _ROWS.setdefault((arm,epoch), []).extend(parent._PROFILE_ROWS.pop((parent.SPATIAL_PROFILE_ARM,epoch), []))


def epoch_audit(arm, epoch):
    rows = _ROWS.pop((arm,epoch), [])
    if not rows:
        raise RuntimeError('missing spatial gradient evidence')
    means = {key:sum(r['branch_gradient_norms'][key] for r in rows)/len(rows)
             for key in rows[0]['branch_gradient_norms']}
    active = epoch > 20
    checks = {
        'finite':all(r['all_combined_gradients_finite'] for r in rows),
        'profile_positive_finite':all(math.isfinite(r['profile_component']) and r['profile_component'] >=0 for r in rows),
        'spatial_path_expected':means['spatial_attention_and_head']>0 if active else means['spatial_attention_and_head']==0,
        'spatial_gate_route':(means['spatial_gate']==0 if not active else means['spatial_gate']>0),
        'detached_base_decoder_zero':means['factor_decoder']==0,
        'detached_agate_parameters_zero':means['agate']==0,
        'no_separate_profile_optimizer_gradient':all(not r['profile_component_separate_optimizer_gradient_added'] for r in rows),
    }
    if not all(checks.values()):
        raise RuntimeError(f'gradient audit {arm}/{epoch}: {checks}')
    return dict(active=active, weight=.10,batch_count=len(rows),checks=checks,
                mean_branch_gradient_norms=means,
                gate_spatial_objective_role='trained')


def singleton_pass(row):
    return bool(row['effective_equals_locked_expression_bitwise']
                and 0 < row['effective_gate_mean'] <= 1
                and 0 < row['effective_gate_max'] <= 1
                and math.isfinite(row['correction_rms'])
                and row['gate_parameter_gradients_finite'])


def technical_map_audit(details):
    weight=details['spatial_map'].detach().float()
    checks={'finite':bool(torch.isfinite(weight).all()),
            'raw_scores_finite':bool(torch.isfinite(details['spatial_raw'].detach()).all()),
            'nonnegative':bool((weight>=0).all()),
            'normalized':bool(torch.allclose(weight.sum(-1),torch.ones_like(weight.sum(-1)),atol=2e-5,rtol=0)),
            'padding_excluded':details['spatial']['padding_selection_count']==0}
    if not all(checks.values()):
        raise RuntimeError(f'Attention validation failed: {checks}')
    return checks


def map_correlation_or_none(weight):
    """Undefined centered correlation is NA, never spurious zero separation."""
    flat=weight.detach().float().permute(1,0,2).reshape(weight.shape[1],-1)
    centered=flat-flat.mean(1,keepdim=True)
    if bool(((flat.amax(1)==flat.amin(1)) | (centered.square().sum(1)==0)).any()):
        return None
    normalized=F.normalize(centered,dim=1)
    corr=normalized@normalized.T
    offdiag=~torch.eye(corr.shape[0],dtype=torch.bool,device=corr.device)
    value=float(corr[offdiag].abs().mean())
    return value if math.isfinite(value) else None


def stage_technical_audit(epoch, phase, minimum_epoch, activation, mechanism):
    """Check gradient routing at each update and cumulative activity after warmup."""
    checks={'common_protocol':activation['common_protocol_audits_pass'] is True,
            'singleton_gate_expression':activation['single_image_gate_gradient_audit']['pass'] is True}
    if phase=='joint':
        checks.update({
            'l2sp_direction_projection_absent':bool(mechanism['l2sp_direction_projection_absent']),
            'l2sp_route_observed':mechanism['l2sp_route_observed_steps']>0,
            'l2sp_route_all_steps_pass':bool(mechanism['l2sp_route_all_steps_pass']),
            'hybrid_route_observed':mechanism['hybrid_route_observed_steps']>0,
            'hybrid_route_all_steps_pass':bool(mechanism['hybrid_route_all_steps_pass']),
            'primary_nonalias':bool(mechanism['plain_vs_agate_nonalias']),
            'forbidden_mae_absent':mechanism['mae_route_observed_steps']==0,
        })
    deferred={'l2sp_positive_after_updates':bool(mechanism['l2sp_positive'])}
    if phase=='joint' and epoch>=minimum_epoch:
        checks.update(deferred)
        checks.update({
            'spatial_gradient_expected':bool(activation['all_factor_spatial_gradients_finite_nonzero']),
            'gate_gradient_expected':bool(activation['all_gate_gradients_finite_nonzero']),
            'spatial_gradient_observed':activation['spatial_gradient_eligible_observed_steps']>0,
            'subbag_gradient_active':activation['subbag_gradient_active_this_epoch'] is True,
            'agate_gradient_observed':mechanism['agate_own_gradient_observed_steps']>0,
            'agate_gradient_all_steps_pass':bool(mechanism['agate_own_gradient_all_steps_finite_nonzero']),
        })
    failures=[name for name,value in checks.items() if not value]
    if failures:
        raise RuntimeError(f'technical audit epoch={epoch} phase={phase} failures={failures}; checks={checks}')
    return dict(checks=checks,deferred_activity=deferred,activity_required_from_epoch=minimum_epoch,
                first_joint_l2sp_zero_allowed=True)


