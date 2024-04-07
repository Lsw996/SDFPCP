import numpy as np


def precision_at_k(y_true, class_probs, k, threshold=0.5, class_of_interest=1, isSorted=False):
    if (not isSorted):
        coi_probs = class_probs[:,class_of_interest]
        sorted_coi_probs = np.sort(coi_probs)[::-1]
        sorted_y = y_true[np.argsort(coi_probs)[::-1]]
    else:
        sorted_coi_probs = class_probs
        sorted_y = y_true

    sorted_coi_probs = sorted_coi_probs[:k]
    sorted_y = sorted_y[:k]
    sorted_predicted_classes = np.where(sorted_coi_probs>threshold,
                                        float(class_of_interest),
                                        0.0)
    precisionK = np.sum(sorted_predicted_classes == sorted_y)/k  
    return precisionK


def map_at_N(y_true, class_probs, N, thrs=0.5, class_of_interest=1):
    pks = []
    coi_probs = class_probs[:, class_of_interest]
    sorted_coi_probs = np.sort(coi_probs)[::-1]
    sorted_y = y_true[np.argsort(coi_probs)[::-1]]
    sorted_coi_probs = sorted_coi_probs[:N]
    sorted_y = sorted_y[:N]

    top_coi_indexes = np.argwhere(sorted_y > 0)

    for value in top_coi_indexes:
        limite = value[0] + 1
        pks.append(
                    precision_at_k(sorted_y[:limite],
                    sorted_coi_probs[:limite],
                    limite, threshold=thrs,isSorted=True)
                    )
    pks = np.array(pks)

    return pks.mean()

