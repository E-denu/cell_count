import numpy as np
import matplotlib.pyplot as plt
from cellpose import models, io, plot, core
from pathlib import Path
import cv2
import gc
from collections import defaultdict
import re
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading



plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Liberation Sans', 'Bitstream Vera Sans'],
    'font.size': 14
})



SAVE_RESULTS = True
SAVE_VISUALIZATIONS = True
MAX_WORKERS = 3

# Per-channel segmentation parameters
CHANNEL_DIAMETERS = {
    'd0': 7,
    'd1': 33, 
    'd3': 7
}

CHANNEL_FLOW_THRESHOLDS = {
    'd0': 0.12,
    'd1': 0.2,
    'd3': 0.1
}


CELLPROB_THRESHOLD = 0.0

save_lock = threading.Lock()

print_lock = threading.Lock()

def safe_print(*args, **kwargs):
    """Thread-safe print function"""
    with print_lock:
        print(*args, **kwargs)


def color_balance(image):
    min_val = 300
    max_val = 3000
    result = np.clip(image, min_val, max_val)
    result = cv2.normalize(result, None, 0, 65535, cv2.NORM_MINMAX).astype(np.uint16)
    return result




def extract_image_base_names(_dir):
    images_dict = defaultdict(list)
    pattern = r'(.+f\d+)(d\d+)(\.TIF)$'
    directory = Path(_dir)
    
    for path in directory.glob('*.TIF'):
        image = path.name
        pattern_match = re.match(pattern, image)
        base = pattern_match.group(1)
        images_dict[base].append((path))
        
    images_dict = dict(sorted(images_dict.items()))
    
    return images_dict


def gpu_status_checker():
    status = core.use_gpu()
    print(f"Device has GPU?: {status}")
    try:
        if status:
            import torch as t
            print(f"Your PyTorch version is: {t.__version__}")
            cuda_status = t.cuda.is_available()
            print(f"CUDA is available?: {cuda_status}")
            if cuda_status:
                gpu_name_version = t.cuda.get_device_name(0)
                print(f"Your GPU device is: {gpu_name_version}")
    except:
        print(f"Your device has no GPU")
    return status


def image_loader(path_to_image):
    image_name_str = str(path_to_image)
    try:
        image = cv2.imread(image_name_str, cv2.IMREAD_UNCHANGED)
    except:
        try:
            image = io.imread(image_name_str)
        except:
            raise ValueError(f"Failed to load image from {path_to_image}")
        
    #image = color_balance(image)
    
    return image



def preprocess_image(image):

    if len(image.shape) == 2:
        return image
    
    elif len(image.shape) == 3:
        if image.shape[0] in [1, 3, 4]:
            image_display = np.transpose(image, (1, 2, 0))
        else:
            image_display = image
        
        if image_display.shape[2] == 1:
            img_display = image_display.squeeze()
        
        elif image_display.shape[2] == 3:
            img_min, img_max = image_display.min(), image_display.max()
            if img_max > img_min:
                image_display = ((image_display - img_min) / 
                              (img_max - img_min) * 255).astype(np.uint8)
        
        elif image_display.shape[2] > 3:
            image_display = image_display[:, :, 0]
        
        return image_display
    
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")




def cell_segmentation(image, model_type, cell_diameter, 
                      flow_threshold=0.1,
                      cellprob_threshold=CELLPROB_THRESHOLD):

    segmentation_results = model_type.eval(
        image, 
        diameter=cell_diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        #channels=channels
    )
    # channels=CHANNELS, 
    # Unpacking segmentation results
    if len(segmentation_results) == 3:
        masks, flows, styles = segmentation_results
    elif len(segmentation_results) == 1:
        masks = segmentation_results[0]
    elif len(segmentation_results) == 2:
        masks = segmentation_results[0]
        flows = segmentation_results[1]
    elif len(segmentation_results) > 2:
        masks = segmentation_results[0]
        flows = segmentation_results[1]
        styles = segmentation_results[2]
    else:
        None
        
    return masks, flows, styles



def cell_counter(masks):
    total_cells = np.max(masks)
    total_cells = int(total_cells)
    return total_cells



def mean_cell_area(masks):
    total_cells = np.max(masks)
    
    if total_cells == 0:
        return 0.0
    
    area_of_cells = []
    
    for cell_id in range(1, total_cells + 1):
        cell_mask = (masks == cell_id)
        area = np.sum(cell_mask)
        if area > 0:
            area_of_cells.append(area)
    
    if len(area_of_cells) > 0:
        return float(np.mean(area_of_cells))
    else:
        return 0.0


def save_masks(masks, output_path):
    with save_lock:
        io.imsave(str(output_path), masks.astype(np.uint16))





def visualize_image(image, masks, total_cells, channel_name, save_path=None):
    
    """Visualize segmentation for a single channel"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    img_display = preprocess_image(image)
    
    if len(img_display.shape) == 2:
        axes[0].imshow(img_display, cmap='gray')
    else:
        axes[0].imshow(img_display)
    
    axes[0].set_title(f'Original Image - {channel_name}', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(masks, cmap='tab20')
    axes[1].set_title(f'Segmentation Masks\n({total_cells} cells detected)', fontsize=14)
    axes[1].axis('off')
    
    if len(img_display.shape) == 2:
        img_for_overlay = img_display
    else:
        img_for_overlay = img_display[:, :, 0]
    
    overlay_min, overlay_max = img_for_overlay.min(), img_for_overlay.max()
    if overlay_max > overlay_min:
        img_normalized = ((img_for_overlay - overlay_min) / 
                         (overlay_max - overlay_min) * 255).astype(np.uint8)
    else:
        img_normalized = np.full_like(img_for_overlay, 128, dtype=np.uint8)
    
    overlay = plot.mask_overlay(img_normalized, masks)
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay', fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    #fig.subplots_adjust(wspace=0.3, hspace=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close(fig)


def write_results_to_csv(csv, row_data):

    dataframe = pd.DataFrame([row_data])
    check_file = csv.exists()
    
    if check_file:
        dataframe.to_csv(csv, mode='a', header=False, index=False)
    else:
        dataframe.to_csv(csv, mode='w', header=True, index=False)




def process_single_channel(path_to_image, channel_name, base_name, model, output_dir):
    
    try:
        safe_print(f"  [{channel_name}] Starting...")
        img = image_loader(path_to_image)
        safe_print("\n  [{0}] Loaded: {1}".format(channel_name, path_to_image.name))
        safe_print("  [{0}] Shape: {1}, dtype: {2}".format(channel_name, img.shape, img.dtype))
        channel_diameter = CHANNEL_DIAMETERS.get(channel_name, 7)
        channel_flow = CHANNEL_FLOW_THRESHOLDS.get(channel_name, 0.1)
        safe_print(f"  [{channel_name}] Parameters: diameter={channel_diameter}, flow_threshold={channel_flow}")
        masks, flows, styles = cell_segmentation(
            img,
            model,
            channel_diameter,
            flow_threshold=channel_flow,
            cellprob_threshold=CELLPROB_THRESHOLD
        )

        # channels=CHANNELS,
        
        total_cells = cell_counter(masks)
        mean_area = mean_cell_area(masks)
        safe_print("\n  [{0}] >>> Detected {1} cells, mean area: {2:.2f} pixels²".format(channel_name,total_cells,mean_area))
        if SAVE_RESULTS and SAVE_VISUALIZATIONS and output_dir:
            viz_path = output_dir / f"{base_name}_{channel_name}_segmentation.png"
            visualize_image(img, masks, total_cells, channel_name, save_path=viz_path)
            mask_path = output_dir / f"{base_name}_{channel_name}_masks.tif"
            save_masks(masks, mask_path)

        del img, masks, flows, styles
        gc.collect()

        return {
            'channel': channel_name,
            'success': True,
            'count': total_cells,
            'mean_area': mean_area,
            'error': None
        }
        
    except Exception as error:
        safe_print(f"  [{channel_name}] ✗ ERROR: {error}")
        import traceback
        traceback.print_exc()
        
        return {
            'channel': channel_name,
            'success': False,
            'count': 0,
            'mean_area': 0.0,
            'error': str(error)
        }




def multithread_all_channels(image_paths, base_name, model, output_base_dir, csv_path):

    safe_print(f"\n{'='*70}")
    safe_print(f"Processing: {base_name}")
    safe_print(f"{'='*70}")

    results_data = {
        'base_name': base_name,
        'd0_count': 0,
        'd1_count': 0,
        'd3_count': 0,
        'd0_mean_area': 0.0,
        'd1_mean_area': 0.0,
        'd3_mean_area': 0.0
    }

    channel_names = ['d0', 'd1', 'd3']

    try:
        if SAVE_RESULTS and SAVE_VISUALIZATIONS:
            output_dir = output_base_dir / base_name
            output_dir.mkdir(exist_ok=True, parents=True)
        else:
            output_dir = None
            
        safe_print(f"\n  Parallel processing of d0, d1 and d3 channels... \n")
        
        channel_results = {}
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_channel = {}
            for img_path, channel_name in zip(image_paths, channel_names):
                future = executor.submit(
                    process_single_channel,
                    img_path,
                    channel_name,
                    base_name,
                    model,
                    output_dir
                )
                future_to_channel[future] = channel_name
            
            # Collect results as they complete
            for future in as_completed(future_to_channel):
                channel_name = future_to_channel[future]
                try:
                    result = future.result()
                    channel_results[channel_name] = result
                except Exception as error:
                    safe_print(f"  [{channel_name}] Exception in thread: {error}")
                    channel_results[channel_name] = {
                        'channel': channel_name,
                        'success': False,
                        'count': 0,
                        'mean_area': 0.0,
                        'error': str(error)
                    }
        
        for channel_name in channel_names:
            result = channel_results.get(channel_name, {
                'success': False,
                'count': 0,
                'mean_area': 0.0
            })
            
            results_data[f'{channel_name}_count'] = result['count']
            results_data[f'{channel_name}_mean_area'] = result['mean_area']

        safe_print(f"\n  Summary:")
        safe_print(f"    d0: {results_data['d0_count']} cells, mean area={results_data['d0_mean_area']:.2f} px²")
        safe_print(f"    d1: {results_data['d1_count']} cells, mean area={results_data['d1_mean_area']:.2f} px²")
        safe_print(f"    d3: {results_data['d3_count']} cells, mean area={results_data['d3_mean_area']:.2f} px² \n")

        write_results_to_csv(csv_path, results_data)
        safe_print(f"  Results appended to {csv_path.name}")

        return {
            'base_name': base_name,
            'success': True,
            'data': results_data,
            'error': None
        }

    except Exception as error:
        safe_print(f" !!! ERROR processing {base_name}: {error}")
        import traceback
        traceback.print_exc()

        write_results_to_csv(csv_path, results_data)

        return {
            'base_name': base_name,
            'success': False,
            'data': results_data,
            'error': str(error)
        }

    finally:
        gc.collect()







