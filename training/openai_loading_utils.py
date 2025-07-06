import argparse
import torch
from training.openai_util import create_model

def parse_args_to_dict(args_string):
    parser = argparse.ArgumentParser()
    for arg in args_string.split('--')[1:]:
        key, value = arg.strip().split(' ', 1)
        parser.add_argument(f'--{key}', type=str, default=value)
    return vars(parser.parse_args([]))

def load_model(state_dict_path, setup_path):
    
    state_dict = torch.load(state_dict_path, map_location=torch.device('cpu'),
                            weights_only=True)
    
    with open(setup_path, 'r') as f:
        args_string = f.read().strip()

    model_args = parse_args_to_dict(args_string)

    # Convert specific arguments to appropriate types
    model_args['attention_resolutions'] = model_args['attention_resolutions']
    model_args['class_cond'] = model_args['class_cond'].lower() == 'true'
    # Drop diffusion_steps and noise_schedule, not necessary for UNetModel
    model_args.pop('diffusion_steps')
    model_args.pop('noise_schedule')
    model_args['dropout'] = float(model_args['dropout'])
    model_args['image_size'] = int(model_args['image_size'])
    model_args['learn_sigma'] = model_args['learn_sigma'].lower() == 'true'
    model_args['num_channels'] = int(model_args['num_channels'])
    model_args['num_head_channels'] = int(model_args['num_head_channels'])
    model_args['num_res_blocks'] = int(model_args['num_res_blocks'])
    model_args['resblock_updown'] = model_args['resblock_updown'].lower() == 'true'
    model_args['use_new_attention_order'] = model_args['use_new_attention_order'].lower() == 'true'
    model_args['use_fp16'] = model_args['use_fp16'].lower() == 'true'
    model_args['use_scale_shift_norm'] = model_args['use_scale_shift_norm'].lower() == 'true'

    model = create_model(**model_args)

    model.load_state_dict(state_dict)
    return model, model_args