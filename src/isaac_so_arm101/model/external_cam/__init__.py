import pickle
import numpy as np

def return_camera_matrix():
    with open("cameraMatrix.pkl", "rb") as f:
        matrix = pickle.load(f)
    return matrix.flatten().tolost()
