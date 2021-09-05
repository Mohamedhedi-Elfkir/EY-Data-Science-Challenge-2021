import numpy as np
import regex as re
from datetime import datetime,timedelta
from skimage.transform import resize


def extract_name(label):
    label = label.split('_')
    for i in range(len(label)):
        if label[i] == "P1":
            break
    return "_".join(label[:i])


def clean_name(name):
    if name is None:
        res = None
    else:
        if name.upper()[-4::] == ".JPG":
            res = name.upper()[:-4].replace(' ', '_')
        else:
            res = name.upper().replace(' ', '_')
    return res


def clean_gdf(label):
    if (label != None):
        label = label.upper().replace(" ", "_").replace("&", "_").replace("(", "_").replace(")", "_").replace(",", "_")
        my_pattern = re.compile(r'\_+')
        label = re.sub(my_pattern, '_', label)
    return label


def extract_id(label):

    '''
    Helper function used to extract the id of an image

    '''

    if (label != None):
        ids = []
        l = label.split("_")
        for part in l:
            try:
                if (len(part) <= 3) and (part != "55"):
                    id = int(part)
                    ids.append(id)
            except:
                continue
        return ids
    else:
        return None


def contain(mask_id, image_id):

    '''
    Match the ID of the linescan with that of the mask

    '''
    if (mask_id != None):
        image_id = image_id[0]
        return image_id in mask_id
    else:
        return False


def get_date(ds):

    '''
    Function to get the date of the fire from the Dataset objects(using the metadoc data)

    ''' 

    date_time_str = ds.metadata_doc['properties']["datetime"].replace("-", "/").replace("T", " ").replace("Z", "")
    fire_date = datetime.strptime(date_time_str, '%Y/%m/%d %H:%M:%S')
    start_date_post = (fire_date + timedelta(days=1))
    end_date_post = (fire_date + timedelta(days=30))
    return start_date_post, end_date_post



def get_coor(linescan):

    '''
    Function to get the coordinates of the fire from the Dataset objects(using the metadoc data)

    '''

    lat = (linescan.metadata_doc["extent"]['lat']["begin"], linescan.metadata_doc["extent"]['lat']["end"])
    long = (linescan.metadata_doc["extent"]['lon']["begin"], linescan.metadata_doc["extent"]['lon']["end"])

    return lat, long

def get_rgb(fire):

    '''
    get_rgb function concetantes the 3 ray channels and RGB, 
    then standarizes the constructed image. So, we end up reducing
    the range of our data from 0-3000 to 0-1 which not only helps 
    us in visualizing the images but also helps during training.

    '''

    red = np.expand_dims(fire["nbar_red"].data, axis=-1)
    red[np.isnan(red)] = 0
    red = (red - red.mean()) / red.std()

    green = np.expand_dims(fire["nbar_green"].data, axis=-1)
    green[np.isnan(green)] = 0
    green = (green - green.mean()) / green.std()

    blue = np.expand_dims(fire["nbar_blue"].data, axis=-1)
    blue[np.isnan(blue)] = 0
    blue = (blue - blue.mean()) / blue.std()

    rgb = np.concatenate((red, green, blue), axis=-1)

    rgb = np.clip(rgb, 0, 1)
    return rgb


def resize_cluster(cluster_X, cluster_Y):

    '''
    resize the cluster from (128,128) to (256,256)

    '''

    n_images = len(cluster_X)
    X = np.zeros((n_images, 256, 256, 3))
    Y = np.zeros((n_images, 256, 256))
    for i in range(len(cluster_X)):
        X[i] = resize(cluster_X[i], (256, 256))
        Y[i] = resize(cluster_Y[i], (256, 256))
    return X, Y
