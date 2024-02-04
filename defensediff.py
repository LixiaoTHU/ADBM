import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
class DefenseDiff(nn.Module):
    def __init__(self, unet, scheduler, classifier, scaler, inv_scaler, 
                 clip = False, grad_checkpoint = True, 
                 noise_level = 100, steps = 10, stack = 1, T=1000):
        super(DefenseDiff, self).__init__()
        self.classifier = classifier
        self.scaler = scaler
        self.inv_scaler = inv_scaler
        self.clip = clip
        self.grad_checkpoint = grad_checkpoint
        if self.grad_checkpoint:
            class SubstituteUnet(torch.nn.Module):
                def __init__(self):
                    super(SubstituteUnet, self).__init__()
                    self.unet = unet

                def forward(self, *args, **kwargs):
                    x = checkpoint(self.unet, *args, **kwargs)
                    return x

            self.unet = SubstituteUnet()
        else:
            self.unet = unet
        self.steps = steps
        self.noise_level = noise_level
        self.scheduler = scheduler
        self.scheduler.set_timesteps(T//(noise_level//steps))
        self.stack = stack
    
    def forward(self, images):
        # ori_images = images.clone()
        images = self.scaler(images)


        for s in range(self.stack):
            noise = torch.randn_like(images, device="cpu").to(images.device)
            previous_noisy_sample = self.scheduler.add_noise(images, noise, 
                                    torch.Tensor([self.noise_level]).long())
            

            # print("the length of -self.steps:",len(self.scheduler.timesteps[-self.steps:]))
            # print(self.scheduler.timesteps[-self.steps:])
            for t in self.scheduler.timesteps[-self.steps:]:
                # t = ft + self.noise_level // self.steps - 1
                ct = t.expand(images.shape[0]).to(images.device)
                noisy_residual = self.unet(previous_noisy_sample, ct)#.sample
                previous_noisy_sample = self.scheduler.step(noisy_residual, t, previous_noisy_sample).prev_sample



            images = previous_noisy_sample

        images = self.inv_scaler(images)
        
        if self.clip:
            images = torch.clamp(images, 0, 1)
        # print(torch.sum((images - ori_images) ** 2, dim=(1, 2, 3)).mean())
        
        output = self.classifier(images)
        return output
