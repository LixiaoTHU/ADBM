import torch
import torch.nn.functional as F
from advutils import get_random_states, set_random_states

class VPSDELoss:
    def __init__(self, tune_T=0.1, 
                 likelihood_weighting=False, 
                 beta_d=19.9, beta_min=0.1, 
                 epsilon_t=1e-5):
        
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t
        self.likelihood_weighting = likelihood_weighting # TD
        self.tune_T = tune_T


    def __call__(self, net, images, labels = None, 
                    t_state = None, z_state=None, adv_images = None, get_x0 = False):

        if t_state:
            state = get_random_states()
            set_random_states(t_state)
            noise_level = self.tune_T if self.tune_T > 0.9 else torch.rand(1).item() * (self.tune_T - 0.1) + 0.1
            # noise_level = self.tune_T
            rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device) \
                            * (noise_level - self.epsilon_t) + self.epsilon_t
            set_random_states(state)
        else:
            noise_level = self.tune_T if self.tune_T > 0.9 else torch.rand(1).item() * (self.tune_T - 0.1) + 0.1
            # noise_level = self.tune_T
            rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device) \
                            * (noise_level - self.epsilon_t) + self.epsilon_t

        if z_state:
            state = get_random_states()
            set_random_states(z_state)
            z = torch.randn_like(images)
            set_random_states(state)
        else:
            z = torch.randn_like(images)
        
        # print("z:", torch.mean(torch.abs(z)))
        # print("t:", torch.mean(torch.abs(rnd_uniform)))
            

        alpha = self.get_alpha(rnd_uniform)

        if adv_images is not None:
            at = alpha
            aT = self.get_alpha(torch.tensor(noise_level, device = images.device))
            inadv = (adv_images - images) * aT * (1 - at) / ((1 - aT) * torch.sqrt(at))
            outadv = (adv_images - images) * torch.sqrt(1/at - 1) * aT / (1 -aT)
        else:
            inadv = torch.zeros_like(images)
            outadv = torch.zeros_like(images)


        images_purb = torch.sqrt(alpha) * images + torch.sqrt(1 - alpha) * z 
        images_purb += inadv
        time_step = (rnd_uniform * 999).squeeze()
        
        # z_pred = net(images_purb, time_cond = time_step, y = labels)
        z_pred = net(images_purb, time_step)


        # add advsarial noise
        z += outadv


        loss = F.mse_loss(z_pred, z)

        if get_x0:
            x0 = (images_purb - torch.sqrt(1 - alpha) * z_pred) / torch.sqrt(alpha)
            return loss, x0
        else:
            return loss


    def get_log_mean_coeff(self, t):
        return -0.25 * t ** 2 * (self.beta_d) - 0.5 * t * self.beta_min

    def get_alpha(self, t):
        return (2 * self.get_log_mean_coeff(t)).exp()