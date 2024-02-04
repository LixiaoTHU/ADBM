import torch
import numpy as np
import random

def get_random_states():
    state = np.random.get_state()
    torch_state = torch.get_rng_state()
    rstate = random.getstate()
    if torch.cuda.is_available():
        cuda_state = torch.cuda.get_rng_state_all()
    else:
        cuda_state = None
    return state, torch_state, rstate, cuda_state

def set_random_states(states):
    state, torch_state, rstate, cuda_state = states
    np.random.set_state(state)
    torch.set_rng_state(torch_state)
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)
    random.setstate(rstate)




def get_adv_batch(model, x_natural, labels, loss_fn, cls_model=None, fix_t=True, fix_z=False, eps = 16, num_steps = 3):
    training = model.training
    model.eval()

    if cls_model:
        cls_model.eval()
        
    if fix_t:
        t_state = get_random_states()
    else:
        t_state = None
    
    x_adv = x_natural.detach() + torch.empty_like(x_natural).uniform_(-eps / 255, eps / 255)
    
    if fix_z:
        z_state = get_random_states()
    else:
        z_state = None

    criterion = torch.nn.CrossEntropyLoss()
    for _ in range(num_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            if cls_model:
                _, x0 = loss_fn(model, x_natural, t_state=t_state, z_state=z_state, adv_images=x_adv, get_x0=True)
                x0 = (x0 + 1) / 2
                pred = cls_model(x0)
                loss = criterion(pred, labels) # guided_loss
            else:
                loss = loss_fn(model, x_natural, t_state=t_state, z_state=z_state, adv_images=x_adv)

        grad = torch.autograd.grad(loss, [x_adv])[0]
        x_adv = x_adv.detach() + (eps / 255) * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - eps / 255), x_natural + eps / 255)
        x_adv = torch.clamp(x_adv, -1.0, 1.0)


    model.training = training
    return x_adv, t_state, z_state
