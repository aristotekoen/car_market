# :car: :greece: AI powered Used Car Price Estimator

## 🔗 https://streamlit-app-305336925991.europe-west1.run.app/ 

### 🤖 How does this work?  
This **machine learning model** estimates the price of a used car based on various features such as **brand, model, mileage, registration year, engine, and more**.  

📊 The model was trained on real **used car listings from Greek websites**, so the predictions reflect the **Greek market**.  

✅ **Simply fill in the details** and click the **Predict** button to get:
- An **estimated price** for your car 💰

Coming soon:
- A **price range** (low & high) 📈
- A **confidence level** for accuracy 🎯
- Analysis of the impact of the main features on your car price


### 🗺️ Context of this project:

Determining the market value of a used car is challenging due to numerous influencing factors. In fact, the price of a used car isn't just a linear function of its mileage and engine size. Other factors such as supply and demand, condition of the car, color, fuel type, interior materials, extras, brand, model, registration year often have a strong impact on a used car's price. Given these complexities, gaining insights from data could help better understand pricing dynamics and help in making more objective estimations of a car's price.  

There are numerous car listing websites on the Greek market and I aim to leverage one of these platforms where sellers list their vehicles characteristics, and buyers can reach out to them.

After a two-year break from data science, this project serves as a hands-on opportunity to refresh and update my skills. The main technical focus areas include:

    Web Scraping & Data Storage – Extracting and storing information from websites via api requests or html code 
    Data Preprocessing & Exploratory Analysis – Extracting meaningful insights from real raw data.
    Machine Learning for Price Prediction – Building and evaluating pricing models.
    Implementing a RAG-based Chatbot – Exploring state-of-the-art retrieval-augmented generation techniques.
    Applying deep learning for feature extraction.



### 🎯 Project Goals

To explore these challenges, I aim to: 

      🕷️ Scrape and store all published used car listings  
      🧹 Clean and preprocess the listings to create a dataset representative of today's Greek market 
      📊 Analyze the data to gain insights and gain a better understanding of the data
      ⚙️ Use machine learning to train a car estimation model, comparing multiple preprocessing and modelling techniques and optimising parameters
      🚀 Deploy the model using a containerized Streamlit app on Google Cloud Run

Next steps: 

      🤖 Chatbot using Retrieval-Augmented Generation (RAG): Enable users to query listings using natural language.
      🔍 Text-Based Behicle Condition Assessment: Fine tune a large language model for sentiment analysis on car listing descriptions
      🖼 Image-Based Feature extraction: Extract Vehicle features from car listing images by fine tuning computer vision transformers to enrich missing fields.
  

