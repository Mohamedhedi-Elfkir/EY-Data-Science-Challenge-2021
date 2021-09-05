import numpy as np
from skimage.transform import rotate
import tensorflow as tf
from metrics import dice_coef_loss, f1_m


def train_models(models, Xs, Ys):
    
    '''
    Training loop for each cluster: Augmenting the data for each cluster
    and feeding it into the corresponding model
    '''

    for i in range(len(models)):

        print("Model ", i + 1, " is Training")
        X = Xs[i]
        Y = Ys[i]
        #     X,Y=resize_cluster(X,Y)
        model = models[i]
        augmented_x = []
        augmented_y = []
        for x, y in zip(X, Y):
            for j in range(30, 360, 30):
                x_trans = rotate(x, j)
                y_trans = rotate(y, j)
                augmented_x.append(x_trans)
                augmented_y.append(y_trans)
        X = np.concatenate((np.array(augmented_x), X), axis=0)
        Y = np.concatenate((np.array(augmented_y), Y), axis=0)
        Y = np.expand_dims(Y, axis=-1)
        callbacks = [tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0, patience=5, verbose=0,
            mode='auto', baseline=None, restore_best_weights=True, ),
            tf.keras.callbacks.ModelCheckpoint(
                "checkpoint", monitor='val_loss', verbose=1, save_best_only=True,
                save_weights_only=False, mode='auto', save_freq='epoch',
                options=None
            )]
        model.fit(X, Y, batch_size=1, epochs=100, validation_split=0.2, callbacks=callbacks)
        models[i] = tf.keras.models.load_model("checkpoint",
                                               custom_objects={"dice_coef_loss": dice_coef_loss, "f1_m": f1_m})
