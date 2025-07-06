
import os
import yaml
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def load_config(folder):
    with open(os.path.join(folder, 'config.yaml'), 'r') as f:
        content = f.read()
        # Remove the EasyDict part
        content = content.split('dictitems:', 1)[-1]
        config = yaml.safe_load(content)
    return config

def get_relevant_folders(path, num_steps):
    relevant_folders = []
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        if not os.path.isdir(folder_path):
            continue
        
        config = load_config(folder_path)
        
        # cond_mech = config.get('conditioning_mechanism')
        # folder_num_steps = config.get('num_steps')
        # image_base_cov = config.get('image_base_covariance')
        # space_step_update_lower = config.get('space_step_update_lower_threshold')
        # cond_scaling = config.get('cond_scaling')
        
        if config.get('num_steps') != num_steps:
            continue
        images_folder = os.path.join(folder_path, 'images')
        if not os.path.exists(images_folder) or not os.listdir(images_folder):
            continue
        relevant_folders.append((folder_path, config))
    
    return relevant_folders

def load_image(folder, subfolder, idx):
    img_path = os.path.join(folder, subfolder, f'00000{idx}_000000.png')
    return Image.open(img_path)

def create_visualization(path, num_steps, idx, method_order=['dps', 'pigdm', 'identity', 'identity_online', 'dct']):
    relevant_folders = get_relevant_folders(path, num_steps)
    
    def sort_and_filter_folders(relevant_folders):
        
        # Create an empty dictionary to store folders for each method
        method_folders = {}
        
        for folder, config in relevant_folders:
            if config['conditioning_mechanism'] == 'dps':
                method_folders['dps'] = (folder, 'DPS')
            elif config['conditioning_mechanism'] == 'pigdm':
                method_folders['pigdm'] = (folder, '$\\pi$GDM')
            elif config['conditioning_mechanism'] == 'tmpd':
                method_folders['tmpd'] = (folder, 'TMPD')
            elif config['conditioning_mechanism'] == 'peng_convert':
                method_folders['peng_convert'] = (folder, 'Peng et al.')
            # elif config['conditioning_mechanism'] == 'online_covariance' and config['image_base_covariance'] == 'identity' and config['space_step_update_lower_threshold'] == 1000:
            #     if config['cond_scaling'] != 1:
            #         method_folders['identity'] = (folder, 'Identity')
            # elif config['conditioning_mechanism'] == 'online_covariance' and config['image_base_covariance'] == 'identity' and config['space_step_update_lower_threshold'] == 1:
            #     if config['cond_scaling'] != 1:
            #         method_folders['identity_online'] = (folder, 'Identity+Online')
            elif config['conditioning_mechanism'] == 'online_covariance' and config['image_base_covariance'] == 'dct_diagonal' and config['cond_scaling'] == 1 and config['denoiser_mean_error_threshold'] == 1.0:
                method_folders['dct'] = (folder, 'FH')
        
        # Define the desired order of methods
        method_order = ['dps', 'pigdm', 'tmpd', 'peng_convert', 'dct']
        
        # Modify this part to keep the full path
        sorted_folders = []
        for method in method_order:
            if method in method_folders:
                sorted_folders.append(method_folders[method])
        
        return sorted_folders

    relevant_folders = sort_and_filter_folders(relevant_folders)
    
    num_methods = len(relevant_folders)
    fig_width = 4 * (num_methods + 2)  # +2 for cond_image and forward_image
    fig = plt.figure(figsize=(fig_width, 4))
    gs = GridSpec(1, num_methods + 2, figure=fig)
    
    # Load and plot cond_image and forward_image (same for all methods)
    cond_img = load_image(relevant_folders[0][0], 'cond_images', idx)
    forward_img = load_image(relevant_folders[0][0], 'forward_images', idx)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(cond_img)
    ax.set_title('Conditional Image')
    ax.axis('off')
    
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(forward_img)
    ax.set_title('Forward Image')
    ax.axis('off')
    
    all_imgs = [cond_img, forward_img]
    
    # Load and plot method-specific images
    imgs = []
    for i, (folder, method) in enumerate(relevant_folders):
        img = load_image(folder, 'images', idx)
        ax = fig.add_subplot(gs[0, i+2])
        ax.imshow(img)
        ax.set_title(f'{method}')
        ax.axis('off')
        imgs.append(img)
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1)
    return fig, cond_img, forward_img, imgs

# set of runs we want to choose here (15/30 steps)
# Conditional image
# Forward image
# DPS -> conditioning_mechanism=dps
# PIGDM -> conditioning_mechanism=pigdm
# TMPD -> conditioning_mechanism=tmpd
# Peng et al. (Convert) -> conditioning_mechanism=peng_convert
# DCT -> conditioning_mechanism=online_covariance, image_base_covariance=dct_diagonal, space_step_update_lower_threshold=1000, cond_scaling=1.0,
# denoiser_mean_error_threshold=1.0

# Example usage
path = f"outputs/generate_conditional/super_resolution_100samples"
operator = path.split("/")[-1]

import argparse

parser = argparse.ArgumentParser(description='Image visualization script')
parser.add_argument('--save_all_images', action='store_true', help='Save all individual images')
parser.add_argument('--operator', type=str, required=True, help='Operator type (e.g., gaussian_blur, inpainting)')
args = parser.parse_args()

# Use the operator from command line arguments
operator = args.operator


# Create the result_images directory if it doesn't exist
result_dir = f'result_images/{operator}'
if os.path.exists(result_dir):
    # Remove all files in the directory
    for filename in os.listdir(result_dir):
        file_path = os.path.join(result_dir, filename)
        if os.path.isfile(file_path):
            os.unlink(file_path)
    print(f"Emptied directory: {result_dir}")
else:
    os.makedirs(result_dir)
    print(f"Created directory: {result_dir}")

for num_steps in [15, 30]:
    for idx in range(0, 5):
        fig, cond_img, forward_img, imgs = create_visualization(path, num_steps, idx)
        fig.savefig(f'result_images/{operator}/{operator}_{num_steps}_{idx}.png')
        plt.close(fig)
        # Save individual images
        image_names = ['conditional', 'forward', 'dps', 'pigdm', 'tmpd', 'peng_convert', 'dct']
        all_images = [cond_img, forward_img] + imgs

        os.makedirs(f'result_images/{operator}/individual', exist_ok=True)

        for img, name in zip(all_images, image_names):
            plt.figure(figsize=(10, 10))
            plt.imshow(img)
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(f'result_images/{operator}/individual/{operator}_{num_steps}_{idx}_{name}.png', bbox_inches='tight', pad_inches=0)
            plt.close()

        print(f"Saved individual images for {operator}, {num_steps} steps, index {idx}")
