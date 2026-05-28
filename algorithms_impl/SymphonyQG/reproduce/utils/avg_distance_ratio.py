import numpy as np 

def calc_ratio(distances, gt_distances):
    ratios = []
    distances = sorted(distances)
    gt_distances = sorted(gt_distances)
    for d1,d2 in zip(distances, gt_distances):
        if d2 < 0.0001:
            continue
        ratios.append(d1/d2)
    if (len(ratios) == 0):
        return 1 * len(gt_distances)
    return sum(ratios) / len(ratios) * len(gt_distances)


def compute_distances(base, query, ids):
    data = base[ids]
    distances = np.linalg.norm(data-query, axis=1)
    return distances