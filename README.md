# :car: :greece: AI powered Used Car Price Estimator

## 🔗 https://streamlit-app-305336925991.europe-west1.run.app/ 

## 🤖 How does this work?  
This **machine learning model** estimates the price of a used car based on various features such as **brand, model, mileage, registration year, engine, and more**.  

📊 The model was trained on real **used car listings from Greek websites**, so the predictions reflect the **Greek market**.  

✅ **Simply fill in the details** and click the **Predict** button to get:
- An **estimated price** for your car 💰

Coming soon:
- A **price range** (low & high) 📈
- A **confidence level** for accuracy 🎯
- Analysis of the impact of the main features on your car price


## 🗺️ Context of this project:

Determining the market value of a used car is challenging due to numerous influencing factors. In fact, the price of a used car isn't just a linear function of its mileage and engine size. Other factors such as supply and demand, condition of the car, color, fuel type, interior materials, extras, brand, model, registration year often have a strong impact on a used car's price. Given these complexities, gaining insights from data could help better understand pricing dynamics and help in making more objective estimations of a car's price.  

There are numerous car listing websites on the Greek market and I aim to leverage one of these platforms where sellers list their vehicles characteristics, and buyers can reach out to them.

After a two-year break from data science, this project serves as a hands-on opportunity to refresh and update my skills. The main technical focus areas include:

    Web Scraping & Data Storage – Extracting and storing information from websites via api requests or html code 
    Data Preprocessing & Exploratory Analysis – Extracting meaningful insights from real raw data.
    Machine Learning for Price Prediction – Building and evaluating pricing models.
    Implementing a RAG-based Chatbot – Exploring state-of-the-art retrieval-augmented generation techniques.
    Applying deep learning for feature extraction.



## 🎯 Project Goals

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
  

## Methodology and Analyses:

### 🕷️ Scraping

The first step consisted in scraping the ads from the chosen website. When looking at the car listing in web developer mode in my browser, the displayed content of the listings was extracted from the response to an api request in the form of a large json file of around 3000 rows with various information about the listing. We can find links to the pictures, informations about the seller, dates of publishing and modifications, extras of the car, main characteristics and many other fields. Below is a screenshot of a small section of such a record:

![image](https://github.com/user-attachments/assets/8ca75dc0-ac6d-4b4d-ae62-f667a13fcbb3)

There are many fields which aren't relevant to our use case, so the first step was to list the keys of the json which interested us. 

Each car listing has an id and we wish to extract all the car listings. So the next step was to find a way to get all the ids on the website. To do this, we use a search on the website without any filter which leads us to the paginated list of all ads. This information was also the result of an api request. For each page of size n_listings, an api request is made to the page number and a json with the n_listings listed on the page are returned. What was useful is that these jsons were much smaller and contained the main characteristics of the listed cars with their ids.

So the methodology to scrape the data was the following: 

* Create an empty dictionnary
* For each page send an api request to get the n_listing listings
* For each listing in the page fill the dictionnary with id as the key and the json with listing information
* Save the final dictionnary as a dataframe

At this stage we get a dataframe with columns containing lists of mixed types or lists of dictionaries (see image below):

![image](https://github.com/user-attachments/assets/c9024264-d3c8-49e3-b0cc-13e3abbdb62e)

We see that the fields are formatted in various ways, some are grouped under dictionnaries with multiple keys, the price is stored as a string with a space between thousands and the EUR sign, mileage is a string with KM sign. That is what we will come to treat in the next section: cleaning and processing. 

Before this however, we then used all the collected ids to scrape the ads one by one to obtain car characterstics which weren't present in the crawling on listing pages such as extras, interior types etc. This allowed us to enrich the features for each listing. 

### 🧹 Cleaning and preprocessing:

This data was very messy both in terms of the way it was stored and in terms of the types (no standard json type for each field within the api response meaning that each field needed to be treated with its own strategy and mixed types within various fields and floats stored as strings with both numbers and letters and special characters). The largest part of the project was spent on this stage and below are the various cleaning methods that we applied to this data.  

#### Cleaning (cf. exploratory_analysis.ipynb): 

✔️ Extracted all possible extras in order to convert extras from a list of jsons each representing the mention of an extra to one boolean column per extra (example: air_conditioning: True or False)  
✔️ Extracted all possible specifications to convert specifications from one json with mentioned specifications to one column per specification containing their respective values (example fuel_type: petrol)  
✔️ Extracted all possible trim levels to convert model trim column from a json with mentioned trim specs to one column per trim spec (example one column doors: number of doors) for each listing  
✔️ Extract description and store it in one columns  
✔️ Deduplicated ads, some listings which were listed with paid priority would appear multiple times on the website so we had to keep only one listing per id  
✔️ We drop over 40 unnecessary columns (leasing, marketplace, seo, finance options etc.)   
✔️ We process geolocation json to extract latitude longitude  
✔️ Convert mileage from a string of the form 15 000 KM to float using regex  
✔️ Extract battery charge time, fuel type, engine power, gearbox type, features, engine size, battery range from key feature dictionnaries  
✔️ Extract Brand, model, variant, registration year from dictionnary  
✔️ Convert engine size from 1200 cc/375kW/1000 W to float  
✔️ Convert dates to datetime  
✔️ Deduplicate redudant columns (some columns were redundant as extracted from different sections of the json)  
✔️ Remove columns with constant values for all listings  
✔️ Convert battery range from 400 km or 400 χλμ to a float via regex  
✔️ Convert battery charge time from 7 ωρες το an integer  
✔️ Translate from Greek to English, Map and Merge redundant category levels for multiple features (fuel type, gearbox_type, interior type, exterior color, interior color etc.). We encountered multiple features such as fuel type where the levels were stored both in English or in Greek within the same column. For example we would have some cars with fuel type Πετρελαιο and some with Diesel. We merged each level together for these columns.   
✔️ Convert year column stored as double digit string in the form 00 for 2000 and 92 for 1992 to the 4 digit year  
✔️ Extract the information Μεταλλικο regarding exterior color to store it as a boolean feature is_metallic  
✔️ Convert emissions Co2 from 98 g/km to float  
✔️ Convert rim size from 17 inches or 17 ιντσες to integer.  
✔️ Fill missing co2 emission values for electric cars to 0  
✔️ Merge inconsistent category levels of body type (Bus with Van and van ) which all consered mini vans  
✔️ Drop listings without a price

We obtain a tabular dataframe with  113877 listings and 145 columns.

#### Preprocessing (cf. 





