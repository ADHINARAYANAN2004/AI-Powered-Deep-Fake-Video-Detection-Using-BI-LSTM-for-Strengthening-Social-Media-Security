from flask import Flask, render_template, request, redirect, flash, session
import torch
from torch import nn
from torchvision import models, transforms
from torch.utils.data import Dataset
import cv2
#import face_recognition
import numpy as np
from werkzeug.utils import secure_filename
import requests
from urllib.parse import urlparse
import os
import sqlite3
import yt_dlp


app = Flask(__name__)
app.secret_key = "7103"

app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'Uploaded_Files')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

DB_FILE = os.path.join(app.root_path, "users.db")
MODEL_PATH = os.path.join(app.root_path, "model", "df_model.pt")


def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        """)
init_db()


class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, batch_first=True, bidirectional=bidirectional)
        self.dp = nn.Dropout(0.4)
        output_dim = hidden_dim * (2 if bidirectional else 1)
        self.linear1 = nn.Linear(output_dim, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))


im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
sm = nn.Softmax(dim=1)

class ValidationDataset(Dataset):
    def __init__(self, video_names, sequence_length=20, transform=None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []
        for frame in self.frame_extract(video_path):
            if frame is None:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.transform is None:
                continue
            try:
                frames.append(self.transform(frame))
            except Exception:
                continue
            if len(frames) == self.count:
                break

        if len(frames) == 0:
            raise ValueError("No valid frames could be extracted from the uploaded video.")

        while len(frames) < self.count:
            frames.append(frames[-1])

        frames = torch.stack(frames)
        return frames.unsqueeze(0)

    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        success = True
        while success:
            success, image = vidObj.read()
            if success:
                yield image


def predict(model, img):
    fmap, logits = model(img)
    logits = sm(logits)
    _, prediction = torch.max(logits, 1)
    confidence = logits[:, int(prediction.item())].item() * 100
    return [int(prediction.item()), confidence]

def detect_fake_video(video_path):
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    video_dataset = ValidationDataset([video_path], sequence_length=20, transform=train_transforms)
    model = Model(2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    prediction = predict(model, video_dataset[0])
    return prediction


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload')
def upload_page():
    if 'user' not in session:
        flash("Please log in to access this page.", "error")
        return redirect('/')
    return render_template('upload.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'user' not in session:
        flash("Please log in to use this feature.", "error")
        return redirect('/')

    temp_video_path = None

    if 'video_file' in request.files and request.files['video_file'].filename != '':
        file = request.files['video_file']
        filename = secure_filename(file.filename)
        temp_video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_video_path)

    elif 'video_url' in request.form and request.form['video_url']:
        video_url = request.form['video_url']
        filename = "downloaded_video.mp4"
        temp_video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        try:
            if video_url.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                response = requests.get(video_url, stream=True, timeout=10)
                if response.status_code == 200:
                    with open(temp_video_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=1024):
                            if chunk:
                                f.write(chunk)
                else:
                    flash("Failed to download direct video file.", "error")
                    return redirect('/upload')
            else:
                ydl_opts = {
                    'outtmpl': temp_video_path,
                    'quiet': True,
                    'format': 'best[ext=mp4]/best'
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([video_url])

            print("Video downloaded successfully:", temp_video_path)
        except Exception as e:
            flash(f"Error downloading video: {str(e)}", "error")
            return redirect('/upload')

    if temp_video_path is None:
        flash("Please upload a video file or provide a video URL.", "error")
        return redirect('/upload')

    try:
        print("Video Path:", temp_video_path)
        print("File size:", os.path.getsize(temp_video_path))
        prediction = detect_fake_video(temp_video_path)
        output = "REAL" if prediction[0] == 1 else "FAKE"
        color = "lime" if output == "REAL" else "red"
        confidence = f"{prediction[1]:.2f}%"
        return render_template("result.html", result=output, confidence=confidence, color=color)
    except Exception as e:
        flash(f"Prediction failed: {str(e)}", "error")
        return redirect('/upload')
    finally:
        if temp_video_path and os.path.exists(temp_video_path):
            os.remove(temp_video_path)


@app.route('/register', methods=['POST'])
def register():
    username = request.form['username']
    email = request.form['email']
    password = request.form['password']
    try:
        with sqlite3.connect(DB_FILE) as conn:
            conn.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", (username, email, password))
        flash("Registration successful! Please login.", "success")
    except sqlite3.IntegrityError:
        flash("Email already exists!", "error")
    return redirect('/')

@app.route('/login', methods=['POST'])
def login():
    email = request.form['login_email']
    password = request.form['login_password']

    with sqlite3.connect(DB_FILE) as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE email = ? AND password = ?", (email, password))
        user = cur.fetchone()

    if user:
        session['user'] = user[1]
        flash(f"Welcome, {user[1]}!", "success")
        return redirect('/upload')
    else:
        flash("Invalid credentials!", "error")
        return redirect('/')

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash("Logged out successfully.", "success")
    return redirect('/')


if __name__ == '__main__':
    app.run(debug=False, port=7103)
