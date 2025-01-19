from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import os

app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        return redirect(url_for('process_file', filename=file.filename))

@app.route('/process/<filename>')
def process_file(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Load the dataset
    df = pd.read_csv(filepath)
    
    # Feature engineering and preprocessing
    features = df[['Energy Consumed (kWh)', 'Charging Duration (hours)', 'Distance Driven (since last charge) (km)']]
    features = features.dropna()  # Drop rows with missing values
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    
    # K-Means clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled_features)
    
    # Map cluster labels to meaningful names
    cluster_labels = {
        0: 'Heavy Users',
        1: 'Light Users',
        2: 'Frequent Users',
        3: 'Moderate Users'
    }
    df['Segment'] = df['Cluster'].map(cluster_labels)
    
    # Save the processed data
    output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + filename)
    df.to_csv(output_filepath, index=False)
    
    # Prepare summary statistics for display
    summary = df.groupby('Segment').agg({
        'Energy Consumed (kWh)': 'mean',
        'Charging Duration (hours)': 'mean',
        'Distance Driven (since last charge) (km)': 'mean',
        'Cluster': 'count'
    }).rename(columns={'Cluster': 'Count'}).reset_index()
    
    return render_template('results.html', tables=[summary.to_html(classes='data', header=True)], file=output_filepath)

@app.route('/download/<filename>')
def download_file(filename):
    return redirect(url_for('static', filename=filename, as_attachment=True))

if __name__ == '__main__':
    app.run(debug=True)
