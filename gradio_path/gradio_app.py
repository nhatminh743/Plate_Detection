import gradio as gr
import requests

BASE_URL = 'http://127.0.0.1:8000/'

def upload_and_return_prediction(file):
    saved_dir = BASE_URL +