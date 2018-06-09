#                               __                    __
#                              /\ \__                /\ \__
#   ___    ___     ___     ____\ \ ,_\    __      ___\ \ ,_\   ____
#  /'___\ / __`\ /' _ `\  /',__\\ \ \/  /'__`\  /' _ `\ \ \/  /',__\
# /\ \__//\ \L\ \/\ \/\ \/\__, `\\ \ \_/\ \L\.\_/\ \/\ \ \ \_/\__, `\
# \ \____\ \____/\ \_\ \_\/\____/ \ \__\ \__/.\_\ \_\ \_\ \__\/\____/
#  \/____/\/___/  \/_/\/_/\/___/   \/__/\/__/\/_/\/_/\/_/\/__/\/___/  .txt
#
#

CASC_PATH = './haarcascade_files/haarcascade_frontalface_default.xml'
SIZE_FACE = 48
EMOTIONS = ['angry', 'disgusted', 'fearful',
            'happy', 'sad', 'surprised', 'neutral']
ANGRY = 0
DISGUSTED =1
FEARFUL = 2
HAPPY = 3
SAD = 4
SURPRISED = 5
NEUTRAL = 6
SAVE_DIRECTORY = './data/'
SAVE_MODEL_FILENAME = 'Gudi_model_10_epochs_2000_faces'
DATASET_CSV_FILENAME = 'fer2013.csv'
SAVE_DATASET_IMAGES_FILENAME = 'data_images.npy'
SAVE_DATASET_LABELS_FILENAME = 'data_labels.npy'
