*Text Emotion Detection Project-

Text Emotion Detection is a web app that predicts the emotion of a text and shows the corresponding emoji.  
It works in real-time using machine learning models.  
Built with python ,scikit-learn, and Streamlit.

	Technologies Used-

 -Python– Programming language  
- scikit-learn– Machine learning models: Logistic Regression, SVM ,Random Forest 
 -pandas– Data handling  
- joblib– Model saving/loading  
- Streamlit– Web interface  
- Altair – Visualization of probabilities  
- NeatText– Text cleaning

	Features-

- Detects emotions: `anger`, `disgust`, `fear`, `happy`, `joy`, `neutral`, `sad`, `shame`, `surprise`  
- Shows  emoji corresponding to detected emotion  
- Displays confidence scores for all emotions.
- Real-time prediction with user-friendly interface.

	Algorithm -

1. Input: User types a text  .
2. Preprocessing: Clean text (remove handles, stopwords).  
3. Feature Extraction: Convert text to vectors using CountVectorizer or TF-IDF .
4. Prediction: Feed vectors into trained ML model  (Logistic Regression / SVM / Random Forest)  .
5. Output: Display predicted emotion , emoji , and confidence.
6. Visualization:  Show probability chart of all emotions using Altair.

How to Run-

1. Install Python (if not installed):
Download Python 3.10+ from [Python Official Website](https://www.python.org/downloads/)

2. Clone the repository:
   
cmd
  git clone https://github.com/arpitapatil03/Text_Emotion_Detection.git
  cd Text_Emotion_Detection

3.Install dependencies:

  pip install -r requirements.txt

4.Run the Streamlit app:

  streamlit run app.py
  
5.Open the app in your browser:
    
  Go to: http://localhost:8501
