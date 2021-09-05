import numpy as np
from skimage.transform import resize
import xarray as xr

def get_results(models, linescantest_df, X_test,shapes,coords,test):

    '''
    Get prediction results on test dataset
    
    '''

    y_pred1 = models[0].predict(X_test[:2])
    y_pred2 = models[1].predict(np.expand_dims(X_test[2], axis=0))
    y_pred3 = models[2].predict(X_test[3:])
    y_pred = np.concatenate((y_pred1, y_pred2, y_pred3), axis=0)
    for i in range(len(linescantest_df)):

        prediction = y_pred[i][:, :, 0]
        if i == 2:
            threshold = 0.45
        elif i == 1:
            threshold = 0.2
        else:
            threshold = 0.7

        #     prediction=Image.fromarray(prediction[:,:,0])
        prediction = resize(prediction, shapes[i])
        #     prediction=np.squeeze(prediction,axis=-1)
        #     prediction=np.expand_dims(prediction,axis=0)
        mask = prediction > threshold
        mask = xr.DataArray(mask, dims=["y", "x"], coords=coords[i])
        # iterate over the coordinates that are required for testing in the current linescan file
        for idx, ob in test.loc[test.label == linescantest_df["label"].iloc[i]].iterrows():
            result_tf = mask.sel(x=ob.x, y=ob.y, method="nearest")
            result_10 = int(result_tf == True)
            test.loc[(test.label == linescantest_df["label"].iloc[i]) & (test.x == ob.x) & (
                    test.y == ob.y), 'target'] = result_10
    return test
