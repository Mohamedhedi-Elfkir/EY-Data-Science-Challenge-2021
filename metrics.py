def dice_coef_loss(y_true, y_pred):
    '''
    Custom building the dice coefficient loss for our segmentation problem.
    Dice Coefficient is 2 * the Area of Overlap divided by the total number of pixels in both images.

    The Dice coefficient is very similar to the IoU. 
    They are positively correlated, meaning if one says model A is better
    than model B at segmenting an image, then the other will say the same. 
    Like the IoU, they both range from 0 to 1, with 1 signifying the 
    greatest similarity between predicted and truth.

    '''
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2 * intersection) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f))


def recall_m(y_true, y_pred):

    '''
    Recall effectively describes the completeness of our positive 
    predictions relative to the ground truth. Of all of the objected 
    annotated in our ground truth, how many did we capture as positive predictions?

    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):

    '''
    Precision effectively describes the purity of our
    positive detections relative to the ground truth.
    Of all of the objects that we predicted in a given image,
    how many of those objects actually had a matching ground truth annotation?

    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):

    '''
    Harmonic mean of the precision and recall
    
    '''

    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
