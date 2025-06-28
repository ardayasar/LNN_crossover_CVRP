import torch

def scores_to_perm(scores):
    """
    Convert 1-D tensor of size n -> permutation list [cust_id, ...].
    Simple argsort descending for first prototype.
    """
    _, order = torch.sort(scores, descending=True)
    # customer IDs start at 1
    perm = (order + 1).tolist()
    return perm