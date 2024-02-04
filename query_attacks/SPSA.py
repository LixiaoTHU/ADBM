import torch

def query_single(model, x, ids, sigma = 0.001, q = 128, dist = "gauss"):
    model.eval()
    g = torch.zeros(x.shape).to(x.device)
    with torch.no_grad():
        for t in range(q):
            if dist == "gauss":
                u = torch.randn(x.shape).to(x.device)
            elif dist == "Rademacher":
                u = torch.empty(x.shape).uniform_(0, 1).to(x.device)
                u = torch.bernoulli(u)
                u = (u - 0.5) * 2
            z1 = model(x + sigma * u)
            z2 = model(x - sigma * u)
            
            Jlist = []
            for i in range(z1.shape[0]):
                z1y = z1[i][ids[i]].clone()
                z1[i][ids[i]] = -10000
                J1 = torch.max(z1[i]) - z1y

                z2y = z2[i][ids[i]].clone()
                z2[i][ids[i]] = -10000
                J2 =  torch.max(z2[i]) - z2y

                Jlist.append(J1 - J2)
            J = torch.Tensor(Jlist).to(x.device)
            J = J.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            g += J * u
    g = g / (q * 2 * sigma)
    return g

class SPSA(object):
    def __init__(self, model, epsilon = 8/255, query_type = "SPSA"):
        self.model = model
        self.model.eval()
        self.epsilon = epsilon
        self.query_type = query_type

    def query(self, imgs, ids):
        adv_imgs = imgs.clone()
        for i in range(40):
            print("step: ", i)
            if self.query_type == "NES":
                est_grad = query_single(self.model, adv_imgs, ids, dist = "gauss")
            elif self.query_type == "SPSA":
                est_grad = query_single(self.model, adv_imgs, ids, dist = "Rademacher")
            adv_imgs = adv_imgs.detach() + (self.epsilon / 8) * est_grad.sign()
            delta = torch.clamp(adv_imgs - imgs, min=-self.epsilon, max=self.epsilon)
            adv_imgs = torch.clamp(imgs + delta, min=0, max=1).detach()
        return adv_imgs
    def __call__(self, imgs, ids):
        return self.query(imgs, ids)