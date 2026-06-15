import numpy as np
from PIL import Image
from .draw_helpers import draw_point, draw_line
from torch.utils.data import DataLoader
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from dataset import MultiChannelSegDataset


def get_cardiothoracic_ratio(npy, img=None, draw=False):
    """
    Calculate Cardio-thoracic-ratio from mask (check https://en.wikipedia.org/wiki/Cardiomegaly)
    MRD = greatest perpendicular diameter from midline to right heart border
    MLD = greatest perpendicular diameter from midline to left heart border
    ID = internal diameter of chest at level of right hemidiaphragm

    Parameters
    ----------
        npy: mask in form of np array [n_classes, width, height]
             The expected order of the `n_classes` axis is:
             1. Lung segmentation mask
             2. Heart segmentation mask
        img: source image, only used for visualization
        draw: whether to visualize the features

    Returns
    -------
        CTR: (MRD + MLD) / ID
    """
    # This label_mapper is for internal use within this function.
    # It assumes the order of masks is fixed as (lung, heart).
    label_mapper = {
        "lung": [0],
        "heart": [1]
    }

    def get_longest(mask):
        min_ = 0
        id_ = 0
        points = (0, 0), (0, 0)
        for i in range(len(mask)):
            if mask[i].sum() == 0:
                continue
            nz = mask[i].nonzero()[0]
            if len(nz) != 0:
                dist = nz.max() - nz.min()
                if dist > min_:
                    min_ = dist
                    id_ = i
                    points = [(i, nz.min()), (i, nz.max())]
        return min_, id_, points

    def distance_to_midline(midline_x, heart):
        mrd = 0
        mld = 0
        mrd_id = 0
        mld_id = 0
        min_pos, max_pos = (0, 0), (0, 0)
        midline1, midline2 = (0, 0), (0, 0)
        for i in range(len(heart)):
            if heart[i].sum() == 0:
                continue
            nz = heart[i].nonzero()[0]
            if len(nz) != 0:
                dist_ld = nz.max() - midline_x
                dist_rd = midline_x - nz.min()

                if dist_rd > mrd:
                    mrd = dist_rd
                    mrd_id = i
                    min_pos = (i, nz.min())
                    midline1 = (i, midline_x)
                if dist_ld > mld:
                    mld = dist_ld
                    mld_id = i
                    max_pos = (i, nz.max())
                    midline2 = (i, midline_x)
        return mrd, mld, mrd_id, mld_id, min_pos, max_pos, midline1, midline2

    if (npy[label_mapper["lung"][0]].sum() == 0) or (npy[label_mapper["heart"][0]].sum() == 0):
        return {
            "score": -1,
            "drawing": Image.new("RGB", (npy.shape[1], npy.shape[2]), "black"),
        }

    lung = npy[label_mapper["lung"][0]]
    
    # Midline is estimated as the horizontal center of the lungs
    lung_cols = np.where(lung.sum(axis=0) > 0)[0]
    if lung_cols.size == 0:
        return {
            "score": -1,
            "drawing": Image.new("RGB", (npy.shape[1], npy.shape[2]), "black"),
        }
    midline = (lung_cols.max() + lung_cols.min()) / 2.0
    
    heart = npy[label_mapper["heart"][0]]

    mrd, mld, _, _, min_pos, max_pos, midline1, midline2 = distance_to_midline(
        midline, heart
    )
    ID, _, points = get_longest(lung)
    
    if ID == 0:
        return {
            "score": -1,
            "drawing": Image.new("RGB", (npy.shape[1], npy.shape[2]), "black"),
        }

    if draw:
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img).convert('RGB')
        
        # NOTE: Assumes draw_point and draw_line are available from the `draw_helpers` module.
        if img is None:
            img = Image.new("RGB", (npy.shape[1], npy.shape[2]), "black")
        else:
            pass

        width = 8

        img = draw_line(
            img,
            (points[0][1], points[0][0]),
            (points[1][1], points[1][0]),
            "#366bc1",
            width,
        )
        img = draw_point(img, (points[0][1], points[0][0]), "#4a6899", width * 2)
        img = draw_point(img, (points[1][1], points[1][0]), "#4a6899", width * 2)

        img = draw_line(
            img, (min_pos[1], min_pos[0]), (midline1[1], midline1[0]), "#d64141", width
        )
        img = draw_line(
            img,
            (midline2[1], midline2[0]),
            (midline1[1], midline1[0]),
            "#d64141",
            width,
        )
        img = draw_line(
            img, (max_pos[1], max_pos[0]), (midline2[1], midline2[0]), "#d64141", width
        )

        img = draw_point(img, (min_pos[1], min_pos[0]), "#e96969", width * 2)
        img = draw_point(img, (max_pos[1], max_pos[0]), "#e96969", width * 2)
        img = draw_point(img, (midline1[1], midline1[0]), "#e96969", width * 2)
        img = draw_point(img, (midline2[1], midline2[0]), "#e96969", width * 2)

        return {"score": (mrd + mld) / ID, "drawing": img}
    return {"score": (mrd + mld) / ID}


def run_ctr_analysis(data_dir: str, draw: bool = False):
    """
    Orchestrates the CTR analysis workflow.

    Args:
        data_dir: Path to the organized dataset directory.
        draw: A flag to determine the return type.
              - If True, returns a dictionary with scores and drawn images.
              - If False, returns a dictionary with scores only.

    Returns:
        A dictionary containing the CTR score for each sample. If `draw` is True,
        the dictionary will also contain the drawn image for each sample.
    """
    selected_classes = [
        "lung_lower_lobe_left",
        "lung_lower_lobe_right",
        "lung_middle_lobe_right",
        "lung_upper_lobe_left",
        "lung_upper_lobe_right",
        "heart"
    ]

    dataset = MultiChannelSegDataset(
        base_dir=data_dir,
        is_train=False,
        selected_views=["PA"],
        selected_classes=selected_classes,
        normalize=False
    )

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    results = {}
    for sample in data_loader:
        # The data loader will now return a tensor with all specified classes.
        combined_masks_np = sample["label"].squeeze(0).permute(0, 2, 1).numpy()
        
        for i, img_np in enumerate(combined_masks_np):
            pil_image = Image.fromarray(img_np.astype(np.uint8))
            
            # 2. Apply the transformations to the PIL Image.
            rotated_img = pil_image.rotate(-90, expand=True)
            flipped_img = rotated_img.transpose(Image.FLIP_LEFT_RIGHT)
            
            # 3. Convert the transformed PIL Image back to a NumPy array.
            transformed_img_np = np.array(flipped_img)
            
            # 4. Put the new transformed array back into the original position.
            combined_masks_np[i] = transformed_img_np

        lung_masks = combined_masks_np[1:6] # The first 5 are the lung lobes
        heart_mask = combined_masks_np[0:1] # The last one is the heart

        combined_lung_mask = np.logical_or.reduce(lung_masks, axis=0)
        
        # Now, stack the combined lung mask and the heart mask.
        final_ctr_input = np.stack([combined_lung_mask, heart_mask.squeeze(0)])

        # Get the original image data
        image_np = sample["data"].squeeze(0).squeeze(0).numpy()
        
        # Calculate the CTR, using the 'draw' flag
        ctr_result = get_cardiothoracic_ratio(final_ctr_input, img=image_np, draw=draw)
        
        sample_id = sample["id"][0]
        
        if draw:
            results[sample_id] = {
                "score": ctr_result["score"],
                "drawing": ctr_result["drawing"]
            }
        else:
            results[sample_id] = ctr_result["score"]
                
    for sample_id, result_data in results.items():
        score = result_data["score"] if draw else result_data
        print(f"ID: {sample_id}, CTR Score: {score:.4f}")
    
    return results

if __name__ == "__main__":
    # Define the path to the organized data
    data_dir = r"/data/zifei/cst_exp/Sample_right_Form"
    run_ctr_analysis(data_dir, draw = True)

