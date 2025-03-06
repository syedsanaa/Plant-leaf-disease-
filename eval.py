import os
import cv2
import py_sod_metrics
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from keras import backend as K

class MyMeanIOU(tf.keras.metrics.MeanIoU):
    '''
    Summary:
        MyMeanIOU inherit tf.keras.metrics.MeanIoU class and modifies update_state function.
        Computes the mean intersection over union metric.
        iou = true_positives / (true_positives + false_positives + false_negatives)
    Arguments:
        num_classes (int): The possible number of labels the prediction task can have
    Return:
        Class objects
    '''

    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(y_true, y_pred, sample_weight)
    
def dice_coef(y_true, y_pred, smooth=1):
    '''
    Summary:
        This functions get dice coefficient metric
    Arguments:
        y_true (float32): true label
        y_pred (float32): predicted label
        smooth (int): smoothness
    Return:
        dice coefficient metric
    '''
    y_true = K.cast(y_true, 'float32') 
    y_pred = K.cast(y_pred, 'float32')
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def jaccard_loss(y_true, y_pred, smooth=1):
    '''
    Summary:
        Computes the Jaccard loss, which is 1 - Jaccard coefficient.
    Arguments:
        y_true (float32): True labels (ground truth).
        y_pred (float32): Predicted labels.
        smooth (int): Smoothing factor to avoid division by zero.
    Returns:
        Jaccard loss value.
    '''
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    jaccard_index = (intersection + smooth) / (union + smooth)
    return 1 - jaccard_index

def dice_coef_score(y_true, y_pred):
    return dice_coef(y_true, y_pred)
def jaccard_loss_score(y_true, y_pred):
    return jaccard_loss(y_true, y_pred)

FM = py_sod_metrics.Fmeasure()
WFM = py_sod_metrics.WeightedFmeasure()
SM = py_sod_metrics.Smeasure()
EM = py_sod_metrics.Emeasure()
MAE = py_sod_metrics.MAE()
MSIOU = py_sod_metrics.MSIoU()

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, required=True, 
                    help="path to the prediction results")
parser.add_argument("--pred_path", type=str, required=True, 
                    help="path to the prediction results")
parser.add_argument("--gt_path", type=str, required=True,
                    help="path to the ground truth masks")
args = parser.parse_args()

"""
def jaccard_index(pred, gt): 
    pred = (pred > 0.5).astype(np.uint8) # Binarize predictions if necessary 
    gt = (gt > 0.5).astype(np.uint8) # Binarize ground truth if necessary 
    intersection = np.logical_and(pred, gt).sum() 
    union = np.logical_or(pred, gt).sum() 
    if union == 0: # Prevent division by zero 
        return 1.0 if intersection == 0 else 0.0 
    return intersection / union

def dice_coefficient(pred, gt): 
    pred = (pred > 0.5).astype(np.uint8) # Binarize predictions if necessary 
    gt = (gt > 0.5).astype(np.uint8) # Binarize ground truth if necessary 
    intersection = np.logical_and(pred, gt).sum() 
    total = pred.sum() + gt.sum() 
    if total == 0: # Prevent division by zero 
        return 1.0 if intersection == 0 else 0.0 
    return 2.0 * intersection / total

def f1_score(pred, gt): 
    pred = (pred > 0.5).astype(np.uint8) # Binarize predictions if necessary 
    gt = (gt > 0.5).astype(np.uint8) # Binarize ground truth if necessary 
    tp = (pred * gt).sum() 
    fp = (pred * (1 - gt)).sum() 
    fn = ((1 - pred) * gt).sum() 
    if tp + fp == 0 or tp + fn == 0: # Prevent division by zero 
        return 0.0 
    precision = tp / (tp + fp) 
    recall = tp / (tp + fn) 
    if precision + recall == 0: # Prevent division by zero 
        return 0.0 
    return 2 * (precision * recall) / (precision + recall)

def mean_iou(pred, gt): 
    pred = (pred > 0.5).astype(np.uint8) # Binarize predictions if necessary 
    gt = (gt > 0.5).astype(np.uint8) # Binarize ground truth if necessary 
    intersection = np.logical_and(pred, gt).sum() 
    union = np.logical_or(pred, gt).sum() 
    if union == 0: # Prevent division by zero 
        return 1.0 if intersection == 0 else 0.0 
    return intersection / union

sample_gray = dict(with_adaptive=True, with_dynamic=True)
sample_bin = dict(with_adaptive=False, with_dynamic=False, with_binary=True, sample_based=True)
overall_bin = dict(with_adaptive=False, with_dynamic=False, with_binary=True, sample_based=False)
FMv2 = py_sod_metrics.FmeasureV2(
    metric_handlers={
        "fm": py_sod_metrics.FmeasureHandler(**sample_gray, beta=0.3),
        "f1": py_sod_metrics.FmeasureHandler(**sample_gray, beta=1),
        "pre": py_sod_metrics.PrecisionHandler(**sample_gray),
        "rec": py_sod_metrics.RecallHandler(**sample_gray),
        "fpr": py_sod_metrics.FPRHandler(**sample_gray),
        "iou": py_sod_metrics.IOUHandler(**sample_gray),
        "dice": py_sod_metrics.DICEHandler(**sample_gray),
        "spec": py_sod_metrics.SpecificityHandler(**sample_gray),
        "ber": py_sod_metrics.BERHandler(**sample_gray),
        "oa": py_sod_metrics.OverallAccuracyHandler(**sample_gray),
        "kappa": py_sod_metrics.KappaHandler(**sample_gray),
        "sample_bifm": py_sod_metrics.FmeasureHandler(**sample_bin, beta=0.3),
        "sample_bif1": py_sod_metrics.FmeasureHandler(**sample_bin, beta=1),
        "sample_bipre": py_sod_metrics.PrecisionHandler(**sample_bin),
        "sample_birec": py_sod_metrics.RecallHandler(**sample_bin),
        "sample_bifpr": py_sod_metrics.FPRHandler(**sample_bin),
        "sample_biiou": py_sod_metrics.IOUHandler(**sample_bin),
        "sample_bidice": py_sod_metrics.DICEHandler(**sample_bin),
        "sample_bispec": py_sod_metrics.SpecificityHandler(**sample_bin),
        "sample_biber": py_sod_metrics.BERHandler(**sample_bin),
        "sample_bioa": py_sod_metrics.OverallAccuracyHandler(**sample_bin),
        "sample_bikappa": py_sod_metrics.KappaHandler(**sample_bin),
        "overall_bifm": py_sod_metrics.FmeasureHandler(**overall_bin, beta=0.3),
        "overall_bif1": py_sod_metrics.FmeasureHandler(**overall_bin, beta=1),
        "overall_bipre": py_sod_metrics.PrecisionHandler(**overall_bin),
        "overall_birec": py_sod_metrics.RecallHandler(**overall_bin),
        "overall_bifpr": py_sod_metrics.FPRHandler(**overall_bin),
        "overall_biiou": py_sod_metrics.IOUHandler(**overall_bin),
        "overall_bidice": py_sod_metrics.DICEHandler(**overall_bin),
        "overall_bispec": py_sod_metrics.SpecificityHandler(**overall_bin),
        "overall_biber": py_sod_metrics.BERHandler(**overall_bin),
        "overall_bioa": py_sod_metrics.OverallAccuracyHandler(**overall_bin),
        "overall_bikappa": py_sod_metrics.KappaHandler(**overall_bin),
    }
)

jaccard_scores = []
dice_scores = []
f1_scores = []
iou_scores = []

pred_root = args.pred_path
mask_root = args.gt_path
mask_name_list = sorted(os.listdir(mask_root))
for i, mask_name in enumerate(mask_name_list):
    print(f"[{i}] Processing {mask_name}...")
    mask_path = os.path.join(mask_root, mask_name)
    pred_path = os.path.join(pred_root, mask_name[:-4] + '.png')
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

    FM.step(pred=pred, gt=mask)
    WFM.step(pred=pred, gt=mask)
    SM.step(pred=pred, gt=mask)
    EM.step(pred=pred, gt=mask)
    MAE.step(pred=pred, gt=mask)
    FMv2.step(pred=pred, gt=mask)

    jaccard = jaccard_index(pred, mask)
    jaccard_scores.append(jaccard)

    dice = dice_coefficient(pred, mask) 
    dice_scores.append(dice)

    f1 = f1_score(pred, mask)
    f1_scores.append(f1)

    iou = mean_iou(pred, mask) 
    iou_scores.append(iou)

mean_jaccard = np.mean(jaccard_scores)
mean_dice = np.mean(dice_scores)
mean_f1 = np.mean(f1_scores)
mean_iou = np.mean(iou_scores)

fm = FM.get_results()["fm"]
wfm = WFM.get_results()["wfm"]
sm = SM.get_results()["sm"]
em = EM.get_results()["em"]
mae = MAE.get_results()["mae"]
fmv2 = FMv2.get_results()

curr_results = {
    #"meandice": fmv2["dice"]["dynamic"].mean(),
    #"meaniou": fmv2["iou"]["dynamic"].mean(),
    #'Smeasure': sm,
    #"wFmeasure": wfm,  # For Marine Animal Segmentation
    #"adpFm": fm["adp"], # For Camouflaged Object Detection
    #"meanEm": em["curve"].mean(),
    #"MAE": mae,
    #"meanDiceCoef": mean_dice,
    #"meanF1Score": mean_f1,
    'MyMeanIOU': m,
    "jaccard": jaccard_loss_score,
    'f1-score': tfa.metrics.F1Score(num_classes=2, average="micro", threshold=0.9),
    'dice_coef_score':dice_coef_score,

}

print(args.dataset_name)
#print("mDice:       ", format(curr_results['meandice'], '.3f'))
#print("mIoU:        ", format(curr_results['meaniou']))
#print("S_{alpha}:   ", format(curr_results['Smeasure'], '.3f'))
#print("F^{w}_{beta}:", format(curr_results['wFmeasure'], '.3f'))
#print("F_{beta}:    ", format(curr_results['adpFm'], '.3f'))
#print("E_{phi}:     ", format(curr_results['meanEm'], '.3f'))
#print("MAE:         ", format(curr_results['MAE'], '.3f'))
print("Jaccard: ", format(curr_results['jaccard'], '.3f'))
print("Dice Coefficient: ", format(curr_results['dice_coef_score'], '.3f'))
print("F1 Score: ", format(curr_results['f1-score'], '.3f'))
print("Mean IoU: ", format(curr_results['MyMeanIOU'], '.3f'))
"""

pred_root = args.pred_path
mask_root = args.gt_path
mask_name_list = sorted(os.listdir(mask_root))
pred_arr = []
mask_arr = []
target_size = (256, 256)
for i, mask_name in enumerate(mask_name_list):
    print(f"[{i}] Processing {mask_name}...")
    mask_path = os.path.join(mask_root, mask_name)
    pred_path = os.path.join(pred_root, mask_name[:-4] + '.png')
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, target_size) 
    pred = cv2.resize(pred, target_size)
    pred_arr.append(pred)
    mask_arr.append(mask)

pred_arr = np.array(pred_arr) 
mask_arr = np.array(mask_arr)

m = MyMeanIOU(2)

print(f"Shape of pred_arr: {pred_arr.shape}")
print(f"Shape of mask_arr: {mask_arr.shape}")

# Assuming pred and gt are your prediction and ground truth arrays
jaccard_score_value = jaccard_loss_score(pred_arr, mask_arr)
dice_score_value = dice_coef_score(pred_arr, mask_arr)
f1_score_value = tfa.metrics.F1Score(num_classes=2, average="micro", threshold=0.9)(pred_arr, mask_arr)
mean_iou_value = m.update_state(pred_arr, mask_arr).result().numpy()

print("Jaccard: ", format(float(jaccard_score_value), '.3f'))
print("Dice Coefficient: ", format(float(dice_score_value), '.3f'))
print("F1 Score: ", format(float(f1_score_value), '.3f'))
print("Mean IoU: ", format(float(mean_iou_value), '.3f'))