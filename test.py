import requests

# Replace 'your_image.jpg' with the actual path to the image file
image_path = './cup.jfif'
files = {'image': open(image_path, 'rb')}

# Replace 'http://127.0.0.1:5000/predict' with the correct URL of your Flask API
url = 'http://127.0.0.1:5000/predict'

# Send a POST request
response = requests.post(url, files=files)

print(response.json())
