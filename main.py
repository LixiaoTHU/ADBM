import argparse, os
import torch
import numpy as np
from accelerate import Accelerator, DistributedDataParallelKwargs

from diffpure import get_unet, get_classifier, ExponentialMovingAverage
from data import get_dataset, get_data_scaler, get_data_inverse_scaler
from utils import *
from loss import VPSDELoss
from advutils import get_adv_batch


def main(jupyter=False):

    def get_args_parser():
        parser = argparse.ArgumentParser('Robust training script', add_help=False)
        parser.add_argument('--configs', default='', type=str)
        parser.add_argument('--unet_ckpt', default="", type=str)

        # these will be updated by configs
        ###########################################
        parser.add_argument('--diff_type', default="sde", choices=["sde", "edm"], type=str)
        parser.add_argument('--cls_version', default="diffpure", type=str)
        parser.add_argument('--resnet_ckpt', default="/home/user/data4/diffusion/DiffPure/WideResNet_70_16_dropout_cfiar10.pth", type=str)
        parser.add_argument('--output_dir', type=str, default="/home/user/data4/diffusion/defensediffusion/test/")


        parser.add_argument('--advtrain', action="store_true", default=False)
        parser.add_argument('--log_freq', type=int, default=100)
        parser.add_argument('--snapshot_freq', type=int, default=5000)
        parser.add_argument('--total_iterations', type=int, default=100000)

        parser.add_argument('--fix_t', action="store_true", default=False)
        parser.add_argument('--fix_z', action="store_true", default=False)
        parser.add_argument('--with_cls', action="store_true", default=False)
        parser.add_argument('--train_cls', action="store_true", default=False)

        parser.add_argument('--tune_T', type=float, default=0.1)
        parser.add_argument('--eps', type=float, default=16)
        ###########################################

        parser.add_argument('--seed', default=42, type=int)

        return parser
    
    
    parser = get_args_parser()
    if jupyter:
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()
    
    args, args_text, config = update_args(args)


    model = get_unet(args, config)

    if args.with_cls:
        cls_model = get_classifier(args.resnet_ckpt, cls_version=args.cls_version, dataset=args.dataset)
        cls_model.eval().requires_grad_(False)
        if args.train_cls:
            cls_model.requires_grad_(True)
    else:
        cls_model = None
    


    set_init_state(args.seed)
    if jupyter:
        return model
    

    # init the dataset
    train_ds, eval_ds = get_dataset(args)
    train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=config.training.batch_size,
      shuffle=True, num_workers=8, drop_last = True)
    eval_dataloader = torch.utils.data.DataLoader(eval_ds, batch_size=config.eval.batch_size,
        shuffle=False, num_workers=8, drop_last = True)
    scaler, inverse_scaler = get_data_scaler(args), get_data_inverse_scaler(args)

    # init the optimizer
    if args.train_cls:
        optimizer = get_optimizer(config, list(model.parameters())+list(cls_model.parameters()))
    else:
        optimizer = get_optimizer(config, model.parameters())

    ema = ExponentialMovingAverage(model.parameters(), config.model.ema_rate)

    # load checkpoint
    state = dict(optimizer=optimizer, model=model, ema=ema, step=0)
    restore_checkpoint(args, state)
    initial_step = int(state['step'])
    print("init step: ", initial_step)
    if initial_step > args.total_iterations:
        initial_step = 0
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs],
        project_dir=os.path.join(args.output_dir, "logs"),
        log_with="tensorboard"
    )
    if accelerator.is_main_process:
        print(f'model_config: {config}')
        print("mix precision: ", accelerator.mixed_precision)
    
    if args.with_cls:
        model, cls_model, optimizer, train_dataloader, eval_dataloader, ema = accelerator.prepare(
            model, cls_model, optimizer, train_dataloader, eval_dataloader, ema)
    else:
        model, optimizer, train_dataloader, eval_dataloader, ema= accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, ema)
    
    set_init_state(args.seed + os.getpid() % args.seed)
    

    VPSDE = VPSDELoss(tune_T=args.tune_T)

    logger = get_logger(args.output_dir)
    if accelerator.is_main_process:
        logger.info(args_text)

    dataloader = (train_dataloader, eval_dataloader)

    train_loop(config, args, model, cls_model, ema, optimizer, initial_step,
                        dataloader, accelerator, VPSDE, scaler, logger)



def validate(args, model, cls_model, eval_dataloader, loss_fn, scaler, accelerator, logger):
    model.eval()
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    for batch_idx, (images, labels) in enumerate(eval_dataloader):
        images = scaler(images)
        if args.advtrain:
            adv_images, t_state, z_state = get_adv_batch(model, images, labels, loss_fn, cls_model=cls_model, fix_t = args.fix_t, fix_z = args.fix_z, eps = args.eps)
            loss, x0 = loss_fn(model, images, t_state = t_state, z_state=z_state, adv_images=adv_images,  get_x0=True)
            if args.train_cls:
                cls_model.eval()
                x0 = (x0 + 1) / 2
                pred = cls_model(x0)
                loss_cls = criterion(pred, labels)
                loss = loss + loss_cls
        else:
            loss = loss_fn(model, images)
        
        loss = accelerator.gather_for_metrics(loss)
        loss = loss.mean()
        
        total_loss += loss.item()
    if accelerator.is_main_process:
        logger.info("eval loss: %s" % (str(total_loss / (batch_idx+1))))



def get_optimizer(config, params):
    """Returns a flax optimizer object based on `config`."""
    if config.optim.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                           weight_decay=config.optim.weight_decay)
    else:
        raise NotImplementedError(
        f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer


def train_loop(config, args, model, cls_model, ema, optimizer, initial_step,
                            dataloader, accelerator, loss_fn, scaler, logger):
    
    total_iterations = args.total_iterations
    state = dict(optimizer=optimizer, model=model, ema=ema, step=initial_step)

    # Initialize an iteration counter
    iteration = initial_step
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    train_dataloader, eval_dataloader = dataloader[0], dataloader[1]

    while iteration < total_iterations:
        for batch_idx, (images, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            images = scaler(images)

            model.train()
            if args.advtrain:
                adv_images, t_state, z_state = get_adv_batch(model, images, labels, loss_fn, cls_model=cls_model, fix_t = args.fix_t, fix_z = args.fix_z)
                optimizer.zero_grad()
                loss, x0 = loss_fn(model, images, t_state = t_state, z_state=z_state, adv_images=adv_images,  get_x0=True)
                if args.train_cls:
                    cls_model.train()
                    x0 = (x0 + 1) / 2
                    pred = cls_model(x0)
                    loss_cls = criterion(pred, labels)
                    loss = loss + loss_cls
            else:
                loss = loss_fn(model, images)

            accelerator.backward(loss)

            if config.optim.warmup > 0:
                for g in optimizer.param_groups:
                    g['lr'] = config.optim.lr * np.minimum((iteration) / config.optim.warmup, 1.0)
            if config.optim.grad_clip >= 0:
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.optim.grad_clip)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=config.optim.grad_clip)
                else:
                    logger.info("no sync")
            optimizer.step()

            ema.update(model)
            
            total_loss += loss.item()

            iteration += 1
            if accelerator.is_main_process and iteration % args.log_freq == 0:
                logger.info("iteration: %s, loss: %s" % (str(iteration), str(total_loss / args.log_freq)))
                total_loss = 0.0
            
            if iteration % args.snapshot_freq == 0:
                accelerator.wait_for_everyone()
                uw_model = accelerator.unwrap_model(model)
                uw_optimizer = accelerator.unwrap_model(optimizer)
                uw_ema = accelerator.unwrap_model(ema)
                if args.train_cls:
                    uw_cls_model = accelerator.unwrap_model(cls_model)
                state = dict(optimizer=uw_optimizer, model=uw_model, 
                                    ema=uw_ema, step=iteration)
                if accelerator.is_main_process:
                    save_checkpoint(os.path.join(args.output_dir, "checkpoint_%s.pth" % (str(iteration))), state)
                    if args.train_cls:
                        torch.save(uw_cls_model.state_dict(), os.path.join(args.output_dir, "checkpoint_cls_%s.pth" % (str(iteration))))

                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                validate(args, model, cls_model, eval_dataloader, loss_fn, scaler, accelerator, logger)
                ema.restore(model.parameters())
            
            if iteration >= total_iterations:
                break
        



if __name__ == "__main__":
    main()
