import os
import symphonyqg
import pandas as pd 
from utils.avg_distance_ratio import calc_ratio, compute_distances
from utils.io import fvecs_read, ivecs_read
from utils.preprocess import normalize
from settings import TOPK, datasets, degrees


if __name__ == "__main__":
    for DATASET in datasets:
        DISTANCE = datasets[DATASET]

        base = fvecs_read(f"./data/{DATASET}/{DATASET}_base.fvecs")
        query = fvecs_read(f"./data/{DATASET}/{DATASET}_query.fvecs")
        gt = ivecs_read(f"./data/{DATASET}/{DATASET}_groundtruth.ivecs")

        NQ, D = query.shape
        N = base.shape[0]

        if DISTANCE == "angular":
            query = normalize(query)

        for DEGREE in degrees[DATASET]:
            index_path = f"./data/{DATASET}/symphonyqg_{DEGREE}.index"
            index = symphonyqg.Index(index_type="QG", metric="L2", num_elements=N, dimension=D, degree_bound=DEGREE)
            index.load(index_path)

            df = pd.read_csv(f"./results/{DATASET}/symphonyqg/symphonyqg{DEGREE}_{TOPK}.csv")
            EFS = list(df['EFS'])
            QPS = list(df['QPS'])
            gtk_distances = [compute_distances(base, query[i], gt[i][:TOPK]) for i in range(NQ)]

            RATIO = []
            for EF in EFS:
                ratios = []
                index.set_ef(EF)
                for i in range(NQ):
                    pred = index.search(query[i], k=TOPK)
                    query_distances = compute_distances(base, query[i], pred) 
                    
                    r = calc_ratio(query_distances, gtk_distances[i])
                    ratios.append(r)
                RATIO.append(sum(ratios) / (TOPK * NQ))
            
            df = pd.DataFrame({"QPS": QPS, "RATIO": RATIO, "EFS": EFS, 
                                        "Method": f"symphonyqg{DEGREE}"})

            res_dir = f"./results/{DATASET}/symphonyqg/"
            try:
                os.makedirs(res_dir)
            except OSError as e:
                print(e)
            df.to_csv(res_dir + f"ratio{DEGREE}_{TOPK}.csv", index=False)