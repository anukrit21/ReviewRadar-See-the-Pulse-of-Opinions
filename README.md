ReviewRadar – See the Pulse of Opinions

ReviewRadar is a web app that analyzes product reviews to tell if they are positive or negative. It helps users and businesses understand what people like or dislike about a product. The app is built using Python and Flask for the backend, and HTML, CSS, and JavaScript for the frontend. It uses a machine learning model to predict the sentiment of reviews and can also find key aspects like quality, delivery, or performance. Reviews in other languages are automatically translated to English before analysis. Users can submit reviews on the website and see sentiment results instantly, along with charts showing overall opinions. This helps businesses improve products and users make better choices.

How to Run ReviewRadar Locally: 
[1] Clone the Repository
    Open a terminal/PowerShell and run:
    git clone https://github.com/anukrit21/ReviewRadar-See-the-Pulse-of-Opinions.git
    cd ReviewRadar-See-the-Pulse-of-Opinions

[2] Create a Virtual Environment
    python -m venv venv

[3] Activate the Virtual Environment
    .\venv\Scripts\Activate.ps1

[4] Install Dependencies
    pip install -r requirements.txt
    (If requirements.txt does not exist, run: pip install flask nltk scikit-learn pandas matplotlib googletrans langdetect flask_sqlalchemy)

[5] Run the App
    python app.py
    Open a web browser and go to:
    http://127.0.0.1:5000

© 2025 Anukrit Sharma. All Rights Reserved.

    
