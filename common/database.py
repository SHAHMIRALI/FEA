import sys
import dropbox
from dropbox.files import WriteMode
from dropbox.exceptions import ApiError, AuthError

from common.credentials import DROPBOX_TOKEN

from common.constants import DATASET_PACKAGE_PATH, TEST_PICS_PACKAGE_PATH

from pprint import pprint

from time import time, gmtime, mktime
from random import random

def auth_dropbox():
    print("Creating a Dropbox object...")
    with dropbox.Dropbox(DROPBOX_TOKEN) as dbx:
        # Check that the access token is valid
        try:
            dbx.users_get_current_account()
        except AuthError:
            sys.exit("ERROR: Invalid access token; try re-generating an "
                "access token from the app console on the web.")
    return dbx

def save_dataset(dbx, dataset_path, dataset_name, description="Dataset: No description."):
    _save_data(dbx, dataset_path, dataset_name, "dataset", description)

def save_test_pics(dbx, test_pics_path, test_pics_name, description="Test pics: No description."):
    _save_data(dbx, test_pics_path, test_pics_name, "test-pics", description)

def _save_data(dbx, data_path, data_name, data_type, description):
    if "_" in data_name:
        raise Exception("Name cannot contain _")
    
    data_upload_path = '/FEA/{0}_{1}_{2}.gzip'.format(int(mktime(gmtime(time()))), data_name, data_type)
    # Uploads contents of LOCALFILE to Dropbox
    with open(data_path, 'rb') as f:
        # We use WriteMode=overwrite to make sure that the settings in the file
        # are changed on upload
        print("Uploading " + data_path + " to Dropbox as " + data_upload_path + "...")
        try:
            dbx.files_upload(f.read(), data_upload_path, mode=WriteMode('overwrite'))
        except ApiError as err:
            # This checks for the specific error where a user doesn't have
            # enough Dropbox space quota to upload this file
            if (err.error.is_path() and
                    err.error.get_path().reason.is_insufficient_space()):
                sys.exit("ERROR: Cannot back up; insufficient space.")
            elif err.user_message_text:
                print(err.user_message_text)
                sys.exit()
            else:
                print(err)
                sys.exit()

def list_data(dbx):
    file_names = map(lambda x: (x.name).split('_'), dbx.files_list_folder('/FEA').entries)
    datasets = list(filter(lambda x: 'dataset' in x[-1], file_names))
    test_pics = list(filter(lambda x: 'test-pics' in x[-1], file_names))

    return datasets, test_pics

if __name__ == "__main__":
    # Check for an access token
    if (len(DROPBOX_TOKEN) == 0):
        sys.exit("ERROR: Looks like you didn't add your access token.")

    dbx = auth_dropbox()
    # save_data(dbx, DATASET_PACKAGE_PATH, "test_auto_deploy", "Initial")
    # save_data(dbx, TEST_PICS_PACKAGE_PATH, "test_auto_deploy", "Initial")