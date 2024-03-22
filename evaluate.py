import argparse, os
import torch

from diffpure import get_unet, get_classifier
from data import get_dataset, get_data_scaler, get_data_inverse_scaler
from utils import *
from defensediff import DefenseDiff
from diffusers import DDPMScheduler, DDIMScheduler, DPMSolverMultistepScheduler
from attack import PGD
from advutils import set_random_states

# from thop import profile

def main():

    def get_args_parser():
        parser = argparse.ArgumentParser('Robust evaluation script', add_help=False)
        parser.add_argument('--configs', default='', type=str)
        parser.add_argument('--unet_ckpt', default="", type=str)

        parser.add_argument('--seed', default=42, type=int)
        parser.add_argument('--gpu', default=1, type=int)
        parser.add_argument('--scheduler', default="ddim", choices=["ddpm", "ddim", "dpmsolver"], type=str)
        parser.add_argument('--steps', default=5, type=int)
        parser.add_argument('--noise_level', default=100, type=int)
        parser.add_argument('--stack', default=1, type=int)

        # attack setting
        parser.add_argument('--attack_type', default="Linf", choices=[ "Linf", "L1", "L2"], help="which attack")
        parser.add_argument('--attack_method', default="none", choices=["none", "PGD", "AA", "RayS", "Square", "SPSA", "test_sigma"], help="attack method")

        parser.add_argument('--eval_eps', default=8, type=int)
        parser.add_argument('--total_step', default=200, type=int)
        parser.add_argument('--rand', action="store_true", default=False)
        parser.add_argument('--capture_random', default=False, action="store_true")
        return parser
    
    
    parser = get_args_parser()
    args = parser.parse_args()
    args, args_text, config = update_args(args)

    model = get_unet(args, config)
    

    classifier = get_classifier(args.resnet_ckpt, cls_version=args.cls_version, dataset=args.dataset)


    print("step:", args.steps)
    set_init_state(args.seed)
    
    
    # init the dataset
    train_ds, eval_ds = get_dataset(args)
    if args.capture_random:
        bs = 32
    else:
        bs = config.eval_attack.batch_size if args.attack_method == "PGD" else config.eval.batch_size
    if args.attack_method in ["RayS", "Square", "SPSA"]:
        bs = 128
    eval_dataloader = torch.utils.data.DataLoader(eval_ds, batch_size= bs * args.gpu,
        shuffle=True, num_workers=8)
    scaler, inverse_scaler = get_data_scaler(args), get_data_inverse_scaler(args)

    # init the scheduler
    if args.scheduler == "ddpm":
        scheduler = DDPMScheduler.from_config("scheduler_configs/ddpm_scheduler_config_revised.json")
    elif args.scheduler == "ddim":
        scheduler = DDIMScheduler.from_config("scheduler_configs/ddpm_scheduler_config_revised.json")
    elif args.scheduler == "dpmsolver":
        scheduler = DPMSolverMultistepScheduler.from_config("scheduler_configs/ddpm_scheduler_config_revised.json")
    
    defensediff = DefenseDiff(model, scheduler, classifier, scaler, inverse_scaler, 
                        clip=True, grad_checkpoint=True, steps=args.steps, 
                        noise_level=args.noise_level, stack=args.stack)


    defensediff.eval().requires_grad_(False)

    # inputs = torch.randn(1, 3, 32, 32)
    # flops, params = profile(defensediff, inputs=(inputs, ))
    # print("Flops: %fG and Params %fM:" % (flops/1e9, params/1e6))


    defensediff = torch.nn.DataParallel(defensediff).cuda()


    epsilon = {"Linf": args.eval_eps / 255, "L2": 1.0, "L1": 12.0}
    step_size = {"Linf": 0.007 if args.eval_eps == 8 else 0.007 / 2, "L2": 0.005, "L1": 1.0}
    if args.attack_method == "none" or args.attack_method == "test_sigma":
        attacker = None
    elif args.attack_method == "PGD":
        ebs = 32 if args.capture_random else 80
        eot_step = 1 if args.capture_random else 20
        attacker = PGD([defensediff], step_size=step_size[args.attack_type], 
                    total_step=args.total_step, eot_step=eot_step, 
                    epsilon=epsilon[args.attack_type], norm=args.attack_type,
                    eot_batch_size=ebs*args.gpu, set_state=args.capture_random)

    elif args.attack_method == "AA":
        from autoattack import AutoAttack
        if args.rand:
            attacker = AutoAttack(defensediff, norm=args.attack_type, eps=epsilon[args.attack_type], version='rand')
        else:
            attacker = AutoAttack(defensediff, norm=args.attack_type, eps=epsilon[args.attack_type], version='standard')

    elif args.attack_method == "RayS":
        from query_attacks import RayS
        attacker = RayS(defensediff, epsilon=epsilon[args.attack_type])
    
    elif args.attack_method == "Square":
        from autoattack import AutoAttack
        attacker = AutoAttack(defensediff, norm=args.attack_type, eps=epsilon[args.attack_type], version='standard')
        attacker.attacks_to_run = ['square']
    
    elif args.attack_method == "SPSA":
        from query_attacks import SPSA
        attacker = SPSA(defensediff, epsilon=epsilon[args.attack_type])




    evaluate(args, config, defensediff, eval_dataloader, attacker)


def evaluate(args, config, model, eval_dataloader, attacker):

    total_acc = 0
    total_adv_acc = 0
    total_num = 0
    total_std = 0
    for i, (images, labels) in enumerate(eval_dataloader):
        images, labels = images.cuda(), labels.cuda()
        if attacker is not None:
            if args.attack_method == "PGD":
                adv_images, state = attacker(images, labels)
                # difference = images - adv_images
                # print(torch.norm(difference.view(difference.shape[0], -1), p=1, dim=1))
            elif args.attack_method == "AA":
                adv_images = attacker.run_standard_evaluation(images, labels, bs=labels.size(0))
            elif args.attack_method == "RayS":
                adv_images, queries, adbd, succ = attacker(images, labels, query_limit=5000)

                images_eps = adv_images - images
                images_eps = torch.clamp(images_eps, -attacker.epsilon, attacker.epsilon)
                adv_images = images + images_eps
            elif args.attack_method == "Square":
                adv_images = attacker.run_standard_evaluation(images, labels, bs=labels.size(0))
            elif args.attack_method == "SPSA":
                adv_images = attacker(images, labels)
                
        with torch.no_grad():
            if args.attack_method == "test_sigma":
                collect_grad = []
                criterion = torch.nn.CrossEntropyLoss()
                with torch.enable_grad():
                    for j in range(10):
                        images.requires_grad = True
                        logits = model(images)
                        if j == 0:
                            preds = torch.argmax(logits, dim=1)
                            acc = (preds == labels).float().mean()
                            total_acc += acc
                        loss = criterion(logits, labels)
                        loss.backward()
                        grad = images.grad.detach().clone().cpu()
                        # grad = grad.reshape(-1).unsqueeze(0)
                        grad = grad.reshape(grad.shape[0], -1).unsqueeze(0)
                        collect_grad.append(grad)
                        images.grad = None
                        images.requires_grad = False
                grad = torch.cat(collect_grad, dim=0) # [10, B, C]
                # std = torch.std(grad, dim=0) # [C]
                # std = std.reshape(images.shape[0], -1).mean(dim=-1) # [B]

                # compute cosine similarity of grad
                grad = grad.permute(1, 0, 2) # [B, 10, C]
                grad = grad / grad.norm(dim=-1, keepdim=True)
                grad = torch.bmm(grad, grad.permute(0, 2, 1)) # [B, 10, 10]
                grad = grad - torch.eye(grad.shape[-1]).unsqueeze(0)
                std = torch.sum(grad, dim=(1, 2)) / 90



                total_std += torch.mean(std)

            else:
                logits = model(images)
                preds = torch.argmax(logits, dim=1)
                acc = (preds == labels).float().mean()
                total_acc += acc

            if attacker is not None:
                if args.capture_random:
                    set_random_states(state)
                adv_logits = model(adv_images)
                adv_preds = torch.argmax(adv_logits, dim=1)
                adv_acc = (adv_preds == labels).float().mean()
                total_adv_acc += adv_acc
            if args.attack_method == "test_sigma":
                print(f"total acc: {total_acc / (i+1)}; total std: {total_std / (i+1)}")
            else:
                print(f"total acc: {total_acc / (i+1)}; total adv acc: {total_adv_acc / (i+1)}")
            total_num += images.shape[0]
            if total_num >= 512:
                break

if __name__ == "__main__":
    main()
