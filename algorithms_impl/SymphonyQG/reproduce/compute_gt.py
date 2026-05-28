import numpy as np
from utils.preprocess import normalize
from utils.io import fvecs_read, ivecs_write

datasets = ["msong"]
dis_type = "l2"

if __name__ == "__main__":
    for DATASET in datasets:
        print(f"Computing groundtruth for {DATASET}")
        base = fvecs_read(f"./data/{DATASET}/{DATASET}_base.fvecs")
        query = fvecs_read(f"./data/{DATASET}/{DATASET}_query.fvecs")

        if dis_type == "angular":
            base = normalize(base)
            query = normalize(query)

        gt = []
        for q in query:
            distances = np.linalg.norm(base - q, axis=1)
            gt.append(np.argsort(distances)[:1000])

        gt = np.array(gt)

        ivecs_write(f"./data/{DATASET}/{DATASET}_groundtruth.ivecs", gt)
