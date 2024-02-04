import torch
# from .parse_config import parse_args_and_config_cifar

from .score_sde.models import utils as scoreutils
# from fvcore.nn import FlopCountAnalysis, flop_count_table



def get_unet(args, config=None):

    # create model
    ckpt = args.unet_ckpt if args.unet_ckpt else args.init_unet_ckpt
    if args.diff_type == 'imagenet':
        # model_config = model_and_diffusion_defaults()
        # model_config.update(vars(config.model))
        # print(f'model_config: {model_config}')
        # model, _ = create_model_and_diffusion(**model_config)
        # if not model_config['class_cond']:
        #     model.load_state_dict(torch.load(f'./resources/checkpoints/'
        #                                      f'DiffPure/256x256_diffusion_uncond.pt'))
        # else:
        #     model.load_state_dict(torch.load('./resources/checkpoints/DiffPure/256x256_diffusion.pt'))
        # if model_config['use_fp16']:
        #     model.convert_to_fp16()
        # img_shape = (3, 256, 256)
        pass

    elif args.diff_type == "sde":

        model = scoreutils.create_model(config)
        if not ckpt:
            print("No checkpoint for unet")
            return model
        state = torch.load(ckpt, map_location='cpu')['ema']['shadow_params']


        parameters = [p for p in model.parameters() if p.requires_grad]
        for s_param, param in zip(state, parameters):
            param.data.copy_(s_param.data)
            
    
    elif args.diff_type == "edm":
        # from .edm import get_edm_cifar_uncond, EDM2VP, SongUNet
        # model = get_edm_cifar_uncond(ckpt)
        # model = EDM2VP(model, device=torch.device("cpu"))
        pass
        
        # model = SongUNet(32, 3, 3)
        # inputs = torch.randn(1, 3, 32, 32)
        # flops = FlopCountAnalysis(model, inputs)
        # print(flop_count_table(flops))
        # # print(model(inputs).shape)
        # exit(0)

    return model


def get_classifier(ckdir, cls_version, dataset):

    
    if "diffpure" in cls_version:
        from .classifiers.diffpure_resnet import WideResNet
        if "diffpure" == cls_version:
            depth = 70
            widen = 16
        else:
            name_parts = cls_version.split('-')
            depth = int(name_parts[1])
            widen = int(name_parts[2])

        model = WideResNet(depth=depth, widen_factor=widen, dropRate=0.3, dataset=dataset)
        if not ckdir:
            return model
        state = torch.load(ckdir, map_location='cpu')
        state = state['ema'] if 'ema' in state else state
        r = {}
        for k, v in list(state.items()):
            if 'module.' in k:
                k = k.split('module.', 1)[1]
                r[k] = v
            else:
                r[k] = v
        model.load_state_dict(r)
        return model
    elif cls_version == "pang": # tmp
        from .classifiers.pang_resnet import wideresnetwithswish
        model = wideresnetwithswish(name="wrn-70-16-swish", dataset=dataset)
        state = torch.load(ckdir, map_location='cpu')
        r = {}
        for k, v in list(state.items()):
            k = k.split('module.0.', 1)[1]
            r[k] = v
        model.load_state_dict(r)
        return model
    elif cls_version == "cifar100_28": # tmp
        from .classifiers.resnet_28 import WideResNet
        model = WideResNet(depth=28, widen_factor=10, num_classes=100, dropRate=0.3)
        state = torch.load(ckdir, map_location='cpu')
        r = {}
        for k, v in list(state['state_dict'].items()):
            if 'module.' in k:
                k = k.split('module.', 1)[1]
                r[k] = v
            else:
                r[k] = v
        model.load_state_dict(r)
        return model
    elif "pang_tiny" in cls_version:
        from .classifiers.pang_tiny_resnet import ti_wideresnetwithswish
        model = ti_wideresnetwithswish(name=cls_version) # e.g., pang_tiny_wrn-70-16-swish
        if not ckdir:
            return model
        
        state = torch.load(ckdir, map_location='cpu')
        state = state['ema'] if 'ema' in state else state
        r = {}
        for k, v in list(state.items()):
            if 'module.' in k:
                k = k.split('module.', 1)[1]
                r[k] = v
            else:
                r[k] = v
        model.load_state_dict(r)
        return model
    else:
        raise NotImplementedError(f"{cls_version} is not implemented")
