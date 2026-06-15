import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import MultiChannelSegDataset
from cardiothoracic_ratio import get_cardiothoracic_ratio
from PIL import Image

def run_ctr_analysis(data_dir: str):
    """
    Orchestrates the CTR analysis workflow.
    
    Args:
        data_dir: Path to the organized dataset directory.
    
    Returns:
        A dictionary containing the CTR score for each sample.
    """
    # The `selected_classes` list is crucial here. The order of these
    # classes determines the order in the `npy` array for the CTR function.
    # We combine the lung lobes by specifying all of them, and then handle
    # the combination inside the data loader.
    selected_classes = [
        "lung_lower_lobe_left",
        "lung_lower_lobe_right",
        "lung_middle_lobe_right",
        "lung_upper_lobe_left",
        "lung_upper_lobe_right",
        "heart"
    ]


    # Initialize the dataset
    dataset = MultiChannelSegDataset(
        base_dir=data_dir,
        is_train=False, # We are not training, just loading
        selected_views=["PA"], # Example view, you may need to change this
        selected_classes=selected_classes
    )

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    results = {}
    for sample in data_loader:
        # The data loader will now return a tensor with all specified classes.
        combined_masks_np = sample["combined_masks_np"].squeeze(0).permute(2, 0, 1).numpy()
        
        # Combine the individual lung lobe masks into a single lung mask.
        # We assume the order matches `selected_classes`.
        lung_masks = combined_masks_np[:5] # The first 5 are the lung lobes
        heart_mask = combined_masks_np[5:6] # The last one is the heart

        # Use a logical OR operation to merge the lung lobes.
        # This creates a single mask where a pixel is "lung" if it belongs
        # to any of the lung lobes.
        combined_lung_mask = np.logical_or.reduce(lung_masks, axis=0)
        
        # Now, stack the combined lung mask and the heart mask.
        final_ctr_input = np.stack([combined_lung_mask, heart_mask.squeeze(0)])
        
        # Get the original image data
        image_np = sample["data"].squeeze(0).squeeze(0).numpy()
        
        # Calculate the CTR
        ctr_result = get_cardiothoracic_ratio(final_ctr_input, img=image_np, draw=False)
        
        sample_id = sample["id"][0]
        results[sample_id] = ctr_result["score"]
        
        print(f"Sample ID: {sample_id}, CTR Score: {ctr_result['score']:.4f}")

    print("\nFinal Results:")
    for sample_id, score in results.items():
        print(f"  - {sample_id}: {score:.4f}")
    
    return results

if __name__ == "__main__":
    # Define the path to the organized data
    data_dir = r"D:\Temp\Sample_right_Form"
    run_ctr_analysis(data_dir)
