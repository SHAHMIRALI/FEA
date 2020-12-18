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

def save_data(dbx, model_path, model_name, description="No description."):
    model_upload_path = '/FEA/{0}_{1}'.format(int(mktime(gmtime(time()))), model_name)
    # Uploads contents of LOCALFILE to Dropbox
    with open(model_path, 'rb') as f:
        # We use WriteMode=overwrite to make sure that the settings in the file
        # are changed on upload
        print("Uploading " + model_path + " to Dropbox as " + model_upload_path + "...")
        try:
            dbx.files_upload(f.read(), model_upload_path, mode=WriteMode('overwrite'))
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

if __name__ == "__main__":
    # Check for an access token
    if (len(DROPBOX_TOKEN) == 0):
        sys.exit("ERROR: Looks like you didn't add your access token.")

    dbx = auth_dropbox()
    save_data(dbx, DATASET_PACKAGE_PATH, "dataset.gzip", "Initial")
    save_data(dbx, TEST_PICS_PACKAGE_PATH, "test_pics.gzip", "Initial")