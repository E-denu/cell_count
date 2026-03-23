from cellpose import models
from pathlib import Path
from package.utils import safe_print, extract_image_base_names
from package.utils import gpu_status_checker, write_results_to_csv, multithread_all_channels
import time

IMAGE_DIRECTORY = '/home/ed488/images/'

OUTPUT_DIR = '/home/ed488/images_results/'
CSV_FILENAME = "cell_counts.csv"
USE_GPU = True
SAVE_RESULTS = True
SAVE_VISUALIZATIONS = False
MODEL_TYPE = 'cpsam'

#cell_type = 'macrophages'
#yeast_typ = 'cryptocus'

def main():
    print("\n"*2)
    print("\U0001F479"*35)
    print("\n                  THE COUNTING DAEMON IS ACTIVATED \n")
    print("\U0001F479"*35)
    print("\n"*2)
    gpu_available = gpu_status_checker()
    if USE_GPU and not gpu_available:
        print("WARNING: GPU requested but not available. Falling back to CPU.")
    
    print(f"\nScanning directory: {IMAGE_DIRECTORY}")
    
    triples = extract_image_base_names(IMAGE_DIRECTORY)
    
    print(f"\nFound {len(triples)} image triples")
    
    if SAVE_RESULTS:
        output_base_dir = Path(OUTPUT_DIR)
        output_base_dir.mkdir(exist_ok=True)
        csv_path = output_base_dir / CSV_FILENAME

        if SAVE_VISUALIZATIONS:
            print(f"Visualizations: Enabled")
        else:
            print(f"Visualizations: Disabled")
        print()
        
        if csv_path.exists():
            csv_path.unlink()
    else:
        output_base_dir = None
        csv_path = None
    
    print(f"Initializing Cellpose model: {MODEL_TYPE}")
    print(f"Using GPU: {USE_GPU and gpu_available}")
    model = models.CellposeModel(gpu=USE_GPU and gpu_available, pretrained_model=MODEL_TYPE)
    print("Model initialized!\n")

    results = []
    processed_count = 0
    error_count = 0
    incomplete_count = 0
    
    print_count = 0
    for key, value in triples.items():
        base_name = key
        image_paths = sorted(value)
        print_count += 1
        
        print(f"\n{'='*70}")
        print(f"Processing {print_count}/{len(triples)}")
        print(f"{'='*70}")
        
        if len(image_paths) == 3:
            result = multithread_all_channels(image_paths, base_name, model, output_base_dir, csv_path)
            results.append(result)
            
            if result['success']:
                processed_count += 1
            else:
                error_count += 1
        else:
            print(f"Warning: {base_name} has {len(image_paths)} images instead of 3")
            for channel, filepath in channel_list:
                print(f"  {channel}: {filepath.name}")
            incomplete_count += 1

            incomplete_data = {
                'base_name': base_name,
                'd0_count': 0,
                'd1_count': 0,
                'd3_count': 0,
                'd0_mean_area': 0.0,
                'd1_mean_area': 0.0,
                'd3_mean_area': 0.0
            }
            if csv_path:
                write_results_to_csv(csv_path, incomplete_data)
            
            results.append({
                'base_name': base_name,
                'success': False,
                'data': incomplete_data,
                'error': 'Incomplete triple'
            })

    print(f"\n{'='*70}")
    print("DAEMON PROCESSING COMPLETE!")
    print(f"{'='*70}")
    print(f"  Successfully processed: {processed_count} triples")
    print(f"  Errors: {error_count} triples")
    print(f"  Incomplete triples: {incomplete_count}")
    print(f"  Total: {len(triples)} triples")
    if csv_path:
        print(f"\n  Results saved to: {csv_path}")
    print(f"{'='*70}")
    
    return results

if __name__ == "__main__":
    script_start = time.time()
    results = main()
    time_taken = time.time() - script_start
    hours, remainder = divmod(int(time_taken), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\n Script completed in {hours:02d}:{minutes:02d}:{seconds:02d} (hh:mm:ss)")
