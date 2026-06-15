from sklearn.linear_model import LinearRegression
import numpy as np
from PIL import Image
from cxas.extraction.func_helpers import get_centers, sort_by_distance, get_min_dist
from cxas.extraction.draw_helpers import draw_point, draw_line
from torch.utils.data import DataLoader
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from dataset import MultiChannelSegDataset

def get_reg_line(points):
        x = np.array([c[0] for c in points])
        y = np.array([c[1] for c in points]).reshape(-1, 1)

        reg = LinearRegression().fit(y, x)

        x_new = np.array(range(512)).reshape(-1, 1)
        y_new = np.array([int(r) for r in reg.predict(x_new)])

        cc = np.array(list(zip(y_new, x_new[:, 0])))

        return cc

def get_spine_center_distance(mask, selected_classes, img=None, draw=False):
    """
    Calculate Spine-Center Distance. Distance from individual vertebrae to a regressed center line from all vertebrae.

    Parameters
    ----------
        mask: mask in form of np array [n_classes, width, height]
        selected_classes: list of class names to process
        img: source image, only used for visualization
        draw: whether to visualize the features

    Returns
    -------
        SCD: Distance from individual vertebrae to a regressed center line from all vertebrae.
    """
    # 1. Record the original size of the mask
    original_height = mask.shape[1]
    original_width = mask.shape[2]
    target_size = (512, 512)
    
    # 2. Resize all masks to 512x512 for calculation
    resized_masks_np = np.zeros((mask.shape[0],) + target_size, dtype=np.uint8)
    for i, m in enumerate(mask):
        pil_mask = Image.fromarray(m.astype(np.uint8))
        resized_mask = pil_mask.resize(target_size, Image.Resampling.NEAREST)
        resized_masks_np[i] = np.array(resized_mask)
    
    vertebrae_indices = list(range(len(selected_classes)))
    centers = get_centers(resized_masks_np, vertebrae_indices)
    centers = [c for c in centers if (c[0] > 0) and (c[1] > 0)]
    
    if not centers:
        return {
            "score": -1,
            "drawing": Image.new("RGB", (mask.shape[1], mask.shape[2]), "black"),
        }
        
    centers = sort_by_distance((256, 0), centers)

    cc = get_reg_line(centers)
    cc_ = [(c[0], c[1]) for c in cc]

    points, center_dists = get_min_dist(centers, cc)

    points = np.array(points)
    cc_ = [c for c in cc_ if (c[1] > centers[0][1]) and (c[1] < centers[-1][1])]

    if draw:
        if isinstance(img, np.ndarray):
            pil_img = Image.fromarray(img).convert('RGB')
            drawing_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
        elif img is not None:
            drawing_img = img.resize(target_size, Image.Resampling.LANCZOS)
        else:
            drawing_img = Image.new("RGB", target_size, "black")

        img = drawing_img

        width = 8


        for idx in range(len(cc_[:-1])):
            img = draw_line(img, cc_[idx], cc_[idx + 1], "#4BC4B6", width)

        for idx in range(len(centers[:-1])):
            img = draw_line(img, centers[idx], centers[idx + 1], "#F4A261", width)

        for idx in range(len(centers)):
            img = draw_line(
                img, centers[idx], (points[idx][0], points[idx][1]), "#4a6899", width
            )

        for idx in range(len(centers)):
            img = draw_point(img, centers[idx], "#d64141", width * 2)

        img = draw_point(img, cc_[0], "#4a6899", width * 2)
        img = draw_point(img, cc_[-1], "#4a6899", width * 2)
        
        final_drawing = drawing_img.resize((original_width, original_height), Image.Resampling.LANCZOS)

        return {"score": np.array(center_dists).mean(), "drawing": final_drawing}
    return {"score": np.array(center_dists).mean()}


def run_scd_analysis(data_dir: str, draw: bool = False):
    """
    Orchestrates the SCD analysis workflow.
    
    Args:
        data_dir: Path to the organized dataset directory.
    
    Returns:
        A dictionary containing the SCD score for each sample.
    """
    selected_classes = [
        "vertebrae_T2", "vertebrae_T3", "vertebrae_T4", "vertebrae_T5", "vertebrae_T6",
        "vertebrae_T7", "vertebrae_T8", "vertebrae_T9", "vertebrae_T10", "vertebrae_T11",
        "vertebrae_T12"
    ]
    print(f"data_dir: {data_dir}")
    dataset = MultiChannelSegDataset(
        base_dir=data_dir,
        is_train=False,
        selected_views=["PA"],
        selected_classes=selected_classes,
        normalize=False, 
        eval_model =True
    )

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    results = {}
    for sample in data_loader:
        # The data loader will now return a tensor with all specified classes.
        combined_masks_np = sample["label"].squeeze(0).permute(0, 2, 1).numpy()
        
        # Apply the transformations to all masks
        for i, img_np in enumerate(combined_masks_np):
            # Convert the NumPy array (a single mask) to a PIL Image.
            # Use .astype(np.uint8) to ensure the data is in the correct format for Pillow.
            pil_image = Image.fromarray(img_np.astype(np.uint8))
            
            # Apply the transformations: rotate -90 degrees and flip left to right.
            rotated_img = pil_image.rotate(-90, expand=True)
            flipped_img = rotated_img.transpose(Image.FLIP_LEFT_RIGHT)
            
            # Convert the transformed PIL Image back to a NumPy array.
            transformed_img_np = np.array(flipped_img)
            
            # Put the new transformed array back into the original position.
            combined_masks_np[i] = transformed_img_np
        
        # Get the original image data
        image_np = sample["data"].squeeze(0).squeeze(0).numpy()
        
        # Pass both the mask and the selected_classes list
        scd_result = get_spine_center_distance(
            combined_masks_np, selected_classes, img=image_np, draw=draw
        )
        
        sample_id = sample["id"][0]
        
        if draw:
            results[sample_id] = {
                "score": scd_result["score"],
                "drawing": scd_result["drawing"]
            }
        else:
            results[sample_id] = scd_result["score"]
        

    for sample_id, result_data in results.items():
        score = result_data["score"] if draw else result_data
        print(f"ID: {sample_id}, SCD Score: {score:.4f}")

    return results


if __name__ == "__main__":
    data_dir = r"/data/zifei/cst_exp/Sample_right_Form"
    run_scd_analysis(data_dir)