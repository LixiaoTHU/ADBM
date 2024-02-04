import argparse, os
import torch
import numpy as np
from accelerate import Accelerator, DistributedDataParallelKwargs

from diffpure import get_classifier  # ExponentialMovingAverage not appliable
from data import get_cls_dataset
from utils import *
from timm.utils import ModelEmaV2


def main():

    def get_args_parser():
        parser = argparse.ArgumentParser('Robust training script', add_help=False)
        parser.add_argument('--configs', default='', type=str)

        # these will be updated by configs
        ###########################################
        parser.add_argument('--cls_version', default="diffpure", type=str)
        parser.add_argument('--resnet_ckpt', default="", type=str)
        parser.add_argument('--restore_ckpt', default="", type=str)
        parser.add_argument('--output_dir', type=str, default="/home/user/data4/diffusion/defensediffusion/test/")

        parser.add_argument('--log_freq', type=int, default=100)
        parser.add_argument('--use_aa', type=bool, default=False)
        ###########################################

        parser.add_argument('--seed', default=42, type=int)

        return parser
    
    
    parser = get_args_parser()
    args = parser.parse_args()
    
    args, args_text, _ = update_args(args)


    cls_model = get_classifier(args.resnet_ckpt, cls_version=args.cls_version, dataset=args.dataset)

    set_init_state(args.seed)
    

    # init the dataset
    train_ds, eval_ds = get_cls_dataset(args)
    train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size,
      shuffle=True, num_workers=8, drop_last = True)
    eval_dataloader = torch.utils.data.DataLoader(eval_ds, batch_size=args.batch_size // 2,
        shuffle=False, num_workers=8, drop_last = True)

    # init the optimizer
    optimizer = get_optimizer(args, cls_model.parameters())

    ema = ModelEmaV2(cls_model, decay= args.ema_rate)

    # load checkpoint
    state = dict(optimizer=optimizer, model=cls_model, ema=ema, step=0)
    restore_cls_checkpoint(args, state)
    initial_epoch = int(state['step'])
    print("init epoch: ", initial_epoch)
    
    ddp_kwargs = DistributedDataParallelKwargs()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs],
        project_dir=os.path.join(args.output_dir, "logs"),
        log_with="tensorboard"
    )
    if accelerator.is_main_process:
        print(f'model_args: {args}')
        print("mix precision: ", accelerator.mixed_precision)
    
    cls_model, optimizer, train_dataloader, eval_dataloader, ema= accelerator.prepare(
        cls_model, optimizer, train_dataloader, eval_dataloader, ema)
    
    set_init_state(args.seed + os.getpid() % args.seed)
    
    

    logger = get_logger(args.output_dir)
    if accelerator.is_main_process:
        logger.info(args_text)

    dataloader = (train_dataloader, eval_dataloader)

    train_loop(args, cls_model, ema, optimizer, initial_epoch,
                        dataloader, accelerator, logger)



def validate(epoch, model, eval_dataloader, accelerator, logger):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    for batch_idx, (images, labels) in enumerate(eval_dataloader):

        with torch.no_grad():
            pred = model(images)

            # for all process
            pred, labels = accelerator.gather_for_metrics((pred, labels))

            loss = criterion(pred, labels)
            total_loss += loss.item()
            total_acc += (pred.argmax(1) == labels).sum().item() / pred.shape[0]



    if accelerator.is_main_process:
        logger.info("epoch: %s, eval loss: %s, eval acc: %s" % (str(epoch), str(total_loss / (batch_idx+1)), str(total_acc / (batch_idx+1))))
    
    return total_acc / (batch_idx+1)




def get_optimizer(args, params):
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    return optimizer

def adjust_learning_rate(args, optimizer, epoch, allepoch):
    if epoch >= 0.5 * allepoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * 0.1
    elif epoch >= 0.75 * allepoch:
        for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.01


def train_loop(args, model, ema, optimizer, initial_epoch,
                            dataloader, accelerator, logger):
    
    total_epochs = args.total_epochs

    # Initialize an iteration counter
    total_loss, total_acc = 0.0, 0.0
    current_acc = 0
    criterion = torch.nn.CrossEntropyLoss()

    train_dataloader, eval_dataloader = dataloader[0], dataloader[1]
    

    iteration = 0
    for epoch in range(initial_epoch, total_epochs):
        adjust_learning_rate(args, optimizer, epoch, allepoch=total_epochs)

        for batch_idx, (images, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()

            model.train()

            pred = model(images)
            loss = criterion(pred, labels)
            acc = (pred.argmax(1) == labels).sum().item()

            accelerator.backward(loss)

            optimizer.step()

            ema.module.update(model)
            
            total_loss += loss.item()
            total_acc += acc / pred.shape[0]

            iteration += 1
            if accelerator.is_main_process and iteration % args.log_freq == 0:
                logger.info("iteration: %s, loss: %s, acc: %s" % (str(iteration), str(total_loss / args.log_freq), str(total_acc / args.log_freq)))
                total_loss = 0.0
                total_acc = 0.0
            
        accelerator.wait_for_everyone()
        uw_model = accelerator.unwrap_model(model)
        uw_optimizer = accelerator.unwrap_model(optimizer)
        uw_ema = accelerator.unwrap_model(ema)
        state = dict(optimizer=uw_optimizer, model=uw_model, 
                            ema=uw_ema, step=epoch)
        with torch.no_grad():
            eval_acc = validate(epoch, ema.module.module, eval_dataloader, accelerator, logger)
        if accelerator.is_main_process:
            if eval_acc > current_acc:
                current_acc = eval_acc
                save_checkpoint(os.path.join(args.output_dir, "checkpoint_best.pth"), state)
            save_checkpoint(os.path.join(args.output_dir, "checkpoint_last.pth"), state)

            
            
        



if __name__ == "__main__":
    main()
