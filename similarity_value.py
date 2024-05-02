import streamlit as st

st.set_page_config(page_title="Autism", layout="wide")

st.subheader("Diagnosing Autism")
st.title("What is Autism?")
st.write("Autism, or Autism Spectrum Disorder (ASD), is a developmental disorder that affects communication, social interaction, and behavior. It's called a spectrum disorder because it affects individuals differently and to varying degrees. Some people with autism may have exceptional abilities in certain areas, while others may struggle with basic tasks.")

with st.container():
    st.write("_____")
    left_column, right_column = st.columns(2)
    with left_column:
        st.title("What is its global pattern?")
        st.write("Autism cases have been increasing drastically over the years which plays a factor in the global development. Though there is increased awareness of its prevalence, there is a higher record of cases in the developing countries. This mostly implies to the lack of health care facilities and a lot more factors.")

with st.container():
    st.write("_____")
    left_column, right_column = st.columns(2)
    with left_column:
        st.title("India and autism")
        st.write("Awareness about autism has been growing in India, leading to improved recognition and diagnosis of the condition. However, there are still significant challenges, particularly in rural areas, where awareness and access to diagnostic services may be limited. Efforts have been made to enhance support services for individuals with autism and their families in India. This includes the development of special education programs, therapy centers, and support groups. However, access to these services can be limited, especially in rural and underserved areas.")

with st.container():
    st.write("_____")
    left_column, right_column = st.columns(2)
    with left_column:
        st.title("What does this website aim for?")
        st.write("This website aims to diagnose autism in children at a basic level and make parents aware of this through an early AI diagnosis and prompt the user to seek medical attention based on the result.")
import cv2
import numpy as np
from scipy.spatial.distance import cosine

def extract_features(video_clip_path):
    cap = cv2.VideoCapture(video_clip_path)
    frame_count = 0
    pixel_sum = 0
    
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        
        pixel_sum += np.mean(frame)
        frame_count += 1
    
    
    average_pixel_intensity = pixel_sum / frame_count
    
    
    cap.release()
    
    return np.array([average_pixel_intensity])


def compare_features(user_features, dataset_features):
    similarities = []
    for dataset_feature in dataset_features:
        similarity = 1 - cosine(user_features, dataset_feature)
        similarities.append(similarity)
    return similarities



user_video_clip =input("Enter the video clip:")
user_features = extract_features(user_video_clip)


dataset_video_clips = ["C:/Users/ADMIN/Desktop/shahid vids/VID_20240428_121803.mp4","C:/Users/ADMIN/Desktop/shahid vids/VID_20240428_121605.mp4",
                       "C:/Users/ADMIN/Desktop/shahid vids/VID_20240428_121620.mp4","C:/Users/ADMIN/Desktop/shahid vids/VID_20240428_121727.mp4",
                       "C:/Users/ADMIN/Desktop/shahid vids/VID_20240428_121756 (1).mp4","C:/Users/ADMIN/Desktop/shahid vids/VID_20240428_121837 (1).mp4",
                       "C:/Users/ADMIN/Desktop/shahid vids/VID_20240428_122814.mp4","C:/Users/ADMIN/Desktop/shahid vids/VID_20240428_122844 (1).mp4"]
dataset_features = [extract_features(clip) for clip in dataset_video_clips]


similarities = compare_features(user_features, dataset_features)


for i, similarity in enumerate(similarities):
    print(f"Similarity with dataset clip {i+1}: {similarity}")
if similarity>0.8:
    print("Autistic")
else:
    print("Not Autistic")
