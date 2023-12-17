import pickle
import webbrowser
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from googleapiclient.discovery import build
import random

import cv2
import numpy as np
from keras.models import model_from_json

# Spotify API Credentials
CLIENT_ID = "6af5f00494464049b17c43bccddc2b45"
CLIENT_SECRET = "668ac2208a474ccfb401bbe4a9b81b4c"

# YouTube API Credentials
YOUTUBE_API_KEY = "AIzaSyD-S7pKIfEpKiQhTe42MmDEmSs6PCWOtiw"

# Initialize the Spotify client
client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Initialize the YouTube API client
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

# Keywords for stress-relief songs
stress_relief_keywords = [
    "relaxing",
    "calm",
    "peaceful",
    "meditative",
    "serene",
    "soothing",
    "tranquil",
    "gentle",
    "mindfulness",
    "zen",
    "chill",
    "ambient",
    "soft",
    "harmonious",
    "comforting",
    "dreamy",
    "stress relief",
    "nature sounds",
    "yoga",
    "meditation"
]

# Function to get YouTube URL for a song
def get_youtube_url(song_name, artist_name):
    search_query = f"{song_name} {artist_name} official audio"
    search_response = youtube.search().list(q=search_query, type='video', part='id', maxResults=1).execute()

    if 'items' in search_response and search_response['items']:
        video_id = search_response['items'][0]['id']['videoId']
        youtube_url = f"https://www.youtube.com/watch?v={video_id}"
        return youtube_url
    else:
        return None


# Function to get the album cover URL for a song
def get_song_album_cover_url(song_name, artist_name):
    search_query = f"{song_name} {artist_name} official audio"
    search_response = youtube.search().list(q=search_query, type='video', part='snippet', maxResults=1).execute()

    if 'items' in search_response and search_response['items']:
        video = search_response['items'][0]
        if 'thumbnails' in video['snippet']:
            thumbnails = video['snippet']['thumbnails']
            # Use the 'medium' thumbnail (you can choose another size if needed)
            album_cover_url = thumbnails['medium']['url']
            return album_cover_url

    # If no thumbnail is found, return a default image URL
    return "https://i.postimg.cc/0QNxYz4V/social.png"

# Function to recommend songs
def recommend(song):
    # Search for stress-relief songs based on keywords
    music_list = []
    for keyword in stress_relief_keywords:
        search_query = f"{keyword} music"
        results = sp.search(q=search_query, type="track", limit=10)
        music_list.extend([(track['name'], track['artists'][0]['name']) for track in results['tracks']['items']])

    # Shuffle the music list for variety
    random.shuffle(music_list)

    # Select the top recommended songs from the shuffled list
    recommended_music_names = []
    recommended_music_urls = []
    recommended_music_posters = []
    for i in range(10):
        song_name, artist_name = music_list[i]
        recommended_music_names.append(song_name)
        recommended_music_urls.append(get_youtube_url(song_name, artist_name))
        recommended_music_posters.append(get_song_album_cover_url(song_name, artist_name))

    return recommended_music_names, recommended_music_urls, recommended_music_posters

def phyPredict(string) :
    strarr = string.split(",")
    arr = [float(ele) for ele in strarr]
    arr = np.array(arr)

    global physicalAtrributesModel
    result = physicalAtrributesModel.predict(arr.reshape(1,-1)).flatten()
    return result[0]


def facePredict(video) :
    if (video=="Stressed") :
      cap = cv2.VideoCapture("Inputs/video2.mp4")
    elif (video=="Not Stressed"):
      cap = cv2.VideoCapture("Inputs/video1.mp4")

    flag = True

    while (flag):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1280, 720))
        if not ret:
            break
        face_detector = cv2.CascadeClassifier('Models/haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

            emotion_prediction = emotion_model.predict(cropped_img)
            maxIndex = int(np.argmax(emotion_prediction))

            if (maxIndex in [0,1,2]):
                cap.release()
                cv2.destroyAllWindows()
                return 1
                flag = False
            else :
                cap.release()
                cv2.destroyAllWindows()
                return 0
                flag = False

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


# Streamlit app
st.header('Driver Stress management')
st.write("Discover soothing music to help reduce stress and promote relaxation.")
phyInput= st.text_input('Physical Attributes Input')

faceInput = st.radio(
    "Emotion Detection Video Input",
    ["Stressed", "Not Stressed"],
    index=None,
)

#selected_song = st.text_input("Enter a song name", "Soothing Music")
#selected_song = st.selectbox(
#    "Type or select a song from the dropdown",
#    stress_relief_keywords
#)



#emotional model load
json_file = open('Models/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights("Models/emotion_model.h5")

#physical attributes model load
physicalAtrributesModel = pickle.load(open('Models/phyattrmodel2.pkl', 'rb'))



#samples
stressed = [-1.20888009e-01,5.09534482e+01,3.72549399e-02,4.70808296e-03,1.41331710e-07,1.54575490e+00,3.11370252e-01,5.10240148e+00,3.40445946e+01,2.17182525e-02,-5.37206037e-04,9.50226244e-02]

notStressed = [-5.38270209e-02,4.11548189e+01,8.43999235e-03,1.35891404e-03,3.79367138e-08,-4.69102221e-01,9.90387794e-02,1.82776374e+00,3.37850909e+01,1.27025734e-02,-2.30913148e-04,1.35669870e-01]


if st.button('Submit'):
  if (phyPredict(phyInput)==2 and facePredict(faceInput)==1):
      recommended_music_names, recommended_music_urls, recommended_music_posters = recommend(random.choice(stress_relief_keywords))
      # Automatically open the top recommended song in a new tab
      if recommended_music_urls:
          webbrowser.open_new_tab(recommended_music_urls[0])

      col1, col2, col3, col4, col5 = st.columns(5)

      for i in range(5):
          with col1 if i == 0 else col2 if i == 1 else col3 if i == 2 else col4 if i == 3 else col5:
              st.text(recommended_music_names[i])
              st.image(recommended_music_posters[i])
              st.success(f"Play Song({recommended_music_urls[i]})")

  else :
    st.text("Driver is Not Stressed")
