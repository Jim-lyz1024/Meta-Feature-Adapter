import torch

def make_optimizer_1stage(cfg, model):
    params = []
    keys = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name} has gradients.")
        else:
            print(f"{name} has no gradients.")
    
    # Add parameters for prompt_learner and text_encoder
    for key, value in model.named_parameters():
        if any(x in key for x in ['prompt_learner', 'text_encoder']):
            lr = cfg.SOLVER.STAGE1.BASE_LR
            weight_decay = cfg.SOLVER.STAGE1.WEIGHT_DECAY
            print(f"Adding parameter {key} to optimizer with lr={lr}")
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
            keys += [key]
            
    # Check if we have any parameters
    if not params:
        raise ValueError("No parameters found for Stage 1 optimization. "
                         "Make sure prompt_learner and text_encoder components exist and are trainable.")
    
    # Create optimizer
    if cfg.SOLVER.STAGE1.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.STAGE1.OPTIMIZER_NAME)(
            params, 
            momentum=cfg.SOLVER.STAGE1.MOMENTUM
        )
    elif cfg.SOLVER.STAGE1.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(
            params,
            lr=cfg.SOLVER.STAGE1.BASE_LR,
            weight_decay=cfg.SOLVER.STAGE1.WEIGHT_DECAY
        )
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.STAGE1.OPTIMIZER_NAME)(params)
    
    return optimizer


def make_optimizer_2stage(cfg, model, center_criterion):
    """
    Create optimizer for stage 2 training - full model training
    """
    params = []
    keys = []
    
    # Process all parameters
    for key, value in model.named_parameters():
        # Skip text encoder parameters
        if "transformer" in key:
            value.requires_grad_(False)
            continue
            
        # Skip meta text generator parameters
        if any(x in key for x in ['meta_adapter.text_generator']):
            value.requires_grad_(False)
            continue
            
        if not value.requires_grad:
            continue
            
        # Set learning rate and weight decay
        lr = cfg.SOLVER.STAGE2.BASE_LR
        weight_decay = cfg.SOLVER.STAGE2.WEIGHT_DECAY
        
        # Adjust bias learning rate
        if "bias" in key:
            lr = cfg.SOLVER.STAGE2.BASE_LR * cfg.SOLVER.STAGE2.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.STAGE2.WEIGHT_DECAY_BIAS
            
        # Adjust classifier learning rate if needed
        if cfg.SOLVER.STAGE2.LARGE_FC_LR:
            if "classifier" in key or "arcface" in key:
                lr = cfg.SOLVER.STAGE2.BASE_LR * 2
                print('Using two times learning rate for fc')
                
        print(f"Adding parameter {key} to optimizer with lr={lr}")
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        keys += [key]
        
    if cfg.SOLVER.STAGE2.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.STAGE2.OPTIMIZER_NAME)(
            params, 
            momentum=cfg.SOLVER.STAGE2.MOMENTUM
        )
    elif cfg.SOLVER.STAGE2.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(
            params,
            lr=cfg.SOLVER.STAGE2.BASE_LR,
            weight_decay=cfg.SOLVER.STAGE2.WEIGHT_DECAY
        )
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.STAGE2.OPTIMIZER_NAME)(params)
        
    optimizer_center = torch.optim.SGD(
        center_criterion.parameters(),
        lr=cfg.SOLVER.STAGE2.CENTER_LR
    )
    
    return optimizer, optimizer_center