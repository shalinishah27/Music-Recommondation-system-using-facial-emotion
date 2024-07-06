import numpy as np
from keras.models import load_model
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.utils import load_img
from fer.fer import FER
import cv2
import os
import google.auth
from googleapiclient.discovery import build
from google.oauth2 import service_account

def sendVideoLinks(emotion):
    # Set up the credentials
    creds = service_account.Credentials.from_service_account_file(r'Path to Json File')
    api_key = os.environ.get('youtube api key')

    # Set up the YouTube Data API client
    youtube = build('youtube', 'v3', credentials=creds)

    # Search for videos that match the query
    search_response = youtube.search().list(
        q=emotion,
        type='video',
        part='id',
        maxResults=10
    ).execute()

    # Extract the video IDs from the search results
    video_ids = []
    for search_result in search_response.get('items', []):
        if search_result['id'].get('videoId'):
            video_ids.append(search_result['id']['videoId'])

    # Construct the YouTube video URLs using the video IDs
    video_urls = []
    for video_id in video_ids:
        video_url = f'https://www.youtube.com/watch?v={video_id}'
        video_urls.append(video_url)
    
    return video_urls


# Read the image in grayscale
img_path = r"C:\Users\home\Desktop\sad.jpg"
img = cv2.imread(img_path)

# Load the trained model
model = load_model(r'C:\Users\home\Desktop\ADT_Project\model.h5')

# Define the input image size
IMG_SIZE = (48, 48)

# Define a list of possible facial emotions
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
img_width, img_height = 48, 48

emotion = "happy"
songsPlayList = sendVideoLinks(emotion)

try:
    img = cv2.imread(r'C:\Users\home\Desktop\sad.jpg')
    detector = FER()
    emotion = detector.detect_emotions(img)
    print(emotion)
    dict = emotion[0]
    emoList = dict['emotions']
    print(emoList)
    emotion = max(emoList, key=emoList.get)
    print('Predicted emotion:', emotion)
    songsPlayList = sendVideoLinks(emotion)
    print(songsPlayList)

except:
    img = cv2.imread(img_path,0)
    img = cv2.resize(img, (img_width, img_height))
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    emotion = emotion_labels[np.argmax(pred)]
    print('Predicted emotion:', emotion)
    songsPlayList = sendVideoLinks(emotion)
    print(songsPlayList)


   


    


