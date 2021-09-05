import numpy as np
from PIL import Image
from utils import contain,get_date,get_coor,get_rgb
import xarray as xr
from skimage.transform import resize
from dea_spatialtools import xr_rasterize
from dea_datahandling import load_ard


def load_linescan_X(dc, df, IMG_HEIGHT, IMG_WIDTH):
    
    '''
    Importing the linescans and putting them in a 4d array

    '''
    X_train = np.zeros((len(df), IMG_HEIGHT, IMG_WIDTH))
    for i in range(len(df)):
        ds = dc.load(product='linescan', id=df['ds'].iloc[i].id, output_crs='epsg:28355',
                     resolution=(-10, 10))
        ds_shape = ds.linescan.data.shape
        im = Image.fromarray(ds.linescan.data.reshape(ds_shape[1], ds_shape[2]))
        im = im.resize((IMG_HEIGHT, IMG_WIDTH))
        im = np.array(im)
        X_train[i] = im

        return X_train

def load_linescan_Y(dc,train,df,gdf,IMG_HEIGHT, IMG_WIDTH):
    y_train = np.zeros((len(train), IMG_HEIGHT, IMG_WIDTH))
    not_assigned = []
    assigned = []
    for i in range(len(train)):
        id = train["id"].iloc[i]
        src = dc.load(product='linescan', id=df["ds"].iloc[i].id, output_crs='epsg:28355',
                      resolution=(-10, 10))
        try:
            ob = gdf.loc[gdf.ids.apply(lambda x: contain(x, id))]
            tgt = xr_rasterize(gdf=ob, da=src)
            im = Image.fromarray(tgt.data)
            im = im.resize((IMG_HEIGHT, IMG_WIDTH))
            im = np.array(im)
            im = (im > 0).astype(int)
            y_train[i] = np.array(im)
            assigned.append(i)
        except:
            not_assigned.append(i)
            continue
    return y_train

def load_satellite_train(dc,df,products,measurements,output_crs,resolution,IMG_HEIGHT, IMG_WIDTH):
    X_train_satellite = np.zeros((len(df), IMG_HEIGHT, IMG_WIDTH, 3))
    found = []
    for i in range(len(df)):
        try:
            ds = df.ds.iloc[i]
            start_date_post, end_date_post = get_date(ds)
            study_area_lat, study_area_lon = get_coor(ds)
            fire = load_ard(dc=dc,
                            products=products,
                            x=study_area_lon,
                            y=study_area_lat,
                            time=(start_date_post, end_date_post),
                            measurements=measurements,
                            min_gooddata=0.5,
                            output_crs=output_crs,
                            resolution=resolution,
                            )
            fire = fire.isel(time=0)
            rgb_img = get_rgb(fire)
            rgb_img = np.clip(rgb_img, 0, 1)
            rgb_img = resize(rgb_img, (IMG_HEIGHT, IMG_WIDTH))

            X_train_satellite[i] = rgb_img
            found.append(i)
        except:
            continue
    return X_train_satellite


# +
# def load_satellite_test(test_labels,linescan_df):
#     #importing test image
#     linescantest_df=linescan_df[linescan_df["label"].apply(lambda x: x in test_labels)]
#     shapes=[]
#     coords=[]
#     X_test=np.zeros((len(linescantest_df),256,256,3))
#     y_test=np.zeros((len(linescantest_df),256,256))

#     for i in range(len(linescantest_df)):
#         try:
#             ds=linescantest_df.ds.iloc[i]
        
#             start_date_post,end_date_post=get_date(ds)
#             study_area_lat,study_area_lon=get_coor(ds)
#             fire = load_ard(dc=dc,
#                     products=products,
#                     x=study_area_lon,
#                     y=study_area_lat,
#                     time=(start_date_post, end_date_post),
#                     measurements=measurements,
#                     min_gooddata=0.5,
#                     output_crs=output_crs,
#                     resolution=resolution,
#                )
#             fire=fire.isel(time=0)
#             rgb_img=get_rgb(fire)
#             rgb_img=np.clip(rgb_img, 0, 1)
#             ds_shape=rgb_img.shape
#             shapes.append(ds_shape[:-1])
#             coords.append(fire.coords)
#             rgb_img=resize(rgb_img,(256,256))
#             X_test[i]=rgb_img
#         except:
#             continue
#     return X_test,shapes,coords
# -

def load_satellite_test(dc,test_labels,linescan_df,products,measurements,output_crs,resolution,IMG_HEIGHT, IMG_WIDTH):
    #importing test image
    linescantest_df=linescan_df[linescan_df["label"].apply(lambda x: x in test_labels)]
    shapes=[]
    coords=[]
    X_test=np.zeros((len(linescantest_df),256,256,3))
    y_test=np.zeros((len(linescantest_df),256,256))

    for i in range(len(linescantest_df)):
        ds=linescantest_df.ds.iloc[i]
        
        start_date_post,end_date_post=get_date(ds)
        study_area_lat,study_area_lon=get_coor(ds)
        fire = load_ard(dc=dc,
                    products=products,
                    x=study_area_lon,
                    y=study_area_lat,
                    time=(start_date_post, end_date_post),
                    measurements=measurements,
                    min_gooddata=0.5,
                    output_crs=output_crs,
                    resolution=resolution,
               )
        fire=fire.isel(time=0)
        rgb_img=get_rgb(fire)
        rgb_img=np.clip(rgb_img, 0, 1)
        ds_shape=rgb_img.shape
        shapes.append(ds_shape[:-1])
        coords.append(fire.coords)
        rgb_img=resize(rgb_img,(256,256))
        X_test[i]=rgb_img
    return X_test,shapes,coords
