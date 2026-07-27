from pathlib import Path
import numpy as np
import nibabel as nib
import cc3d
from tqdm import tqdm


BACKGROUND = 0
ED = 1
NC = 2
ET = 3

# ==========================================================
# REMOVE SMALL CONNECTED COMPONENTS
# ==========================================================

def remove_small_components(seg, min_size=50):
    """
    Remove connected regions smaller than min_size voxels
    independently for ED, NC, ET.
    """

    new_seg = np.zeros_like(seg)

    for label in [ED, NC, ET]:

        binary = (seg == label).astype(np.uint8)

        cc, n_cc = cc3d.connected_components(
            binary,
            connectivity=26,
            return_N=True
        )

        for i in range(1, n_cc + 1):

            component = (cc == i)

            if component.sum() >= min_size:
                new_seg[component] = label

    return new_seg


# ==========================================================
# RATIOS
# ==========================================================

def get_wt_voxels(seg):
    return np.sum(seg != 0)


def get_ed_ratio(seg):

    wt = get_wt_voxels(seg)

    if wt == 0:
        return 0

    return np.sum(seg == ED) / wt


def get_nc_ratio(seg):

    wt = get_wt_voxels(seg)

    if wt == 0:
        return 0

    return np.sum(seg == NC) / wt


def get_et_ratio(seg):

    wt = get_wt_voxels(seg)

    if wt == 0:
        return 0

    return np.sum(seg == ET) / wt


# ==========================================================
# LABEL REDEFINITION
# ==========================================================

def suppress_small_et(seg, threshold=0.04):
    """
    ET -> NC if ET/WT is too small
    """

    ratio = get_et_ratio(seg)

    if ratio < threshold:
        seg[seg == ET] = NC

    return seg


def suppress_small_ed(seg, threshold=1.0):
    """
    ED -> NC if ED/WT is too small
    """

    ratio = get_ed_ratio(seg)

    if ratio < threshold:
        seg[seg == ED] = NC

    return seg


# ==========================================================
# SINGLE CASE POSTPROCESS
# ==========================================================

def postprocess_case(
        seg,
        component_threshold=50,
        ed_ratio_threshold=None,
        et_ratio_threshold=0.04
):

    # -------------------------
    # Step 1:
    # Remove tiny regions
    # -------------------------

    seg = remove_small_components(
        seg,
        min_size=component_threshold
    )

    # -------------------------
    # Step 2:
    # ET correction
    # -------------------------

    if et_ratio_threshold is not None:
        seg = suppress_small_et(
            seg,
            threshold=et_ratio_threshold
        )

    # -------------------------
    # Step 3:
    # ED correction
    # -------------------------

    if ed_ratio_threshold is not None:
        seg = suppress_small_ed(
            seg,
            threshold=ed_ratio_threshold
        )

    return seg


# ==========================================================
# DIRECTORY POSTPROCESS
# ==========================================================

def postprocess_directory(
        input_dir,
        output_dir,
        component_threshold=50,
        et_ratio_threshold=0.04,
        ed_ratio_threshold=None
):

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    files = sorted(input_dir.glob("*.nii.gz"))

    print()
    print("=" * 60)
    print(f"Found {len(files)} predictions")
    print("=" * 60)

    for file in tqdm(files):

        nii = nib.load(str(file))

        seg = nii.get_fdata().astype(np.uint8)

        seg_pp = postprocess_case(
            seg,
            component_threshold=component_threshold,
            et_ratio_threshold=et_ratio_threshold,
            ed_ratio_threshold=ed_ratio_threshold
        )

        nib.save(
            nib.Nifti1Image(
                seg_pp.astype(np.uint8),
                nii.affine,
                nii.header
            ),
            output_dir / file.name
        )

    print()
    print("Done!")
    print("Saved to:", output_dir)


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    INPUT_DIR = PROJECT_ROOT / "test_predictions_DA"

    OUTPUT_DIR = PROJECT_ROOT / "test_predictions_DA_post"

    postprocess_directory(
        INPUT_DIR,
        OUTPUT_DIR,

        # remove tiny blobs
        component_threshold=50,

        # Top-1 PED setting
        et_ratio_threshold=0.04,

        # disable ED correction
        ed_ratio_threshold=None
    )