import os, logging, yaml, datetime
import random
import numpy as np
import torch
import argparse
from diffpure import dict2namespace

def get_outdir(path, *paths, inc=False):
    outdir = os.path.join(path, *paths)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    elif inc:
        count = 1
        outdir_inc = outdir + '-' + str(count)
        while os.path.exists(outdir_inc):
            count = count + 1
            outdir_inc = outdir + '-' + str(count)
            assert count < 100
        outdir = outdir_inc
        os.makedirs(outdir)
    print("outdir: ", outdir)
    return outdir


def set_init_state(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)

    # set cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def save_checkpoint(ckpt_dir, state):
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': state['model'].state_dict(),
        'ema': state['ema'].state_dict(),
        'step': state['step']
    }
    torch.save(saved_state, ckpt_dir)

def restore_checkpoint(args, state):
    ckpt_dir = args.unet_ckpt if args.unet_ckpt else args.init_unet_ckpt
    if not ckpt_dir:
        print("No checkpoint for restoring")
        return
    loaded_state = torch.load(ckpt_dir, map_location='cpu')


    model_dict = dict()
    try:
        for n, v in loaded_state['model'].items():
            name = n.split("module.")[-1]
            model_dict[name] = v
        state['model'].load_state_dict(model_dict, strict=False)
    except:
        print("model load failed")

    try:
        state['optimizer'].load_state_dict(loaded_state['optimizer'])
    except:
        print("optimizer load failed")
        
    try:
        state['ema'].load_state_dict(loaded_state['ema'])
    except:
        print("ema load failed")

    try:
        state['step'] = loaded_state['step']
    except:
        print("step load failed")


def restore_cls_checkpoint(args, state):
    ckpt_dir = args.restore_ckpt
    if not ckpt_dir:
        print("No checkpoint for restoring")
        return
    loaded_state = torch.load(ckpt_dir, map_location='cpu')


    model_dict = dict()
    for n, v in loaded_state['model'].items():
        name = n.split("module.")[-1]
        model_dict[name] = v
    state['model'].load_state_dict(model_dict)

    state['optimizer'].load_state_dict(loaded_state['optimizer'])
        
    state['ema'].load_state_dict(loaded_state['ema'])

    state['step'] = loaded_state['step']

def get_logger(file_path):
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')

    fh = logging.FileHandler(os.path.join(file_path, '%s.txt' % (datetime.datetime.now().strftime("%m-%d-%H%M%S"))))
    fh.setLevel('INFO')
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel('INFO')
    ch.setFormatter(formatter)


    logger = logging.getLogger()
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.setLevel('INFO')
    return logger

def update_args(args):
    opt = vars(args)
    if args.configs:
        opt.update(yaml.load(open(args.configs), Loader=yaml.FullLoader))
        if args.other_config:
            with open(args.other_config, 'r') as f:
                config = yaml.safe_load(f)
            config = dict2namespace(config)
        else:
            config = None
            worn = "No other config is provided"
            print(worn)
    else:
        print("configs is a must")
        exit(0)
    
    args = argparse.Namespace(**opt)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text, config