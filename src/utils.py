import numpy as np

def get_bounding_boxes(gt_map: np.array):
    '''A function to get the bounding boxes given an input mask.
    
    Args:
        - gt_map: np.array
    '''
    y_idx, x_idx = np.where(gt_map > 0) # returns the index where there is a mask
    x_min, x_max = np.min(x_idx), np.max(x_idx)
    y_min, y_max = np.min(y_idx), np.max(y_idx)

    # mask the bounding box bigger such that, it completely encloses the object
    H, W = gt_map.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = max(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = max(H, y_max + np.random.randint(0, 20))
    bbox = [x_min, y_min, x_max, y_max]
    return bbox


def print_trainable_params(model):
    """
    Prints, number of trainable parameters
    """
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100*trainable_params/all_params:.3f}%"
    )
