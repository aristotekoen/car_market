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

### 🧹 Cleaning (cf. exploratory_analysis.ipynb):

This data was very messy both in terms of the way it was stored and in terms of the types (no standard json type for each field within the api response meaning that each field needed to be treated with its own strategy and mixed types within various fields and floats stored as strings with both numbers and letters and special characters). The largest part of the project was spent on this stage and below are the various cleaning methods that we applied to this data.  


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

We obtain a tabular dataframe with  112944 listings and 145 columns.

### 📊 Exploratory Analysis (cf.5_exploratory_analysis_after_initial_clean.ipynb): 

#### 1 - Missing values: 

We notice some features with very low level of completion <50% such as vehicle dimensions, performance features such as torque, technical check up, battery related information and co2 emissions. Also, we see the extra features all with the same level of completion, this is because when the json containing extras was present we set to true the extras contained in it and to false the ones not mentionned. Therefore all extras are set to NaN for listings which did not contain a json with extras.

![completude](https://github.com/user-attachments/assets/1682b5b6-d858-4eee-a2a9-d1e1a45549c2)

#### 2 - Analysis of fields extracted from extras: 

The json response from car ads contains a json with the mentioned extras in the car. We have 85 possible extras for each car, that is a very large number of features and we want to see if certain extras are really useful or present in the dataframe. 

We notice that certain extras are almost never set to True such as foreign numbers, price without vat. We also see some extras that are almost always set to True such as abs. We notice extra:826 whith a non informative name, we will not include this in our dataset.

![extras](https://github.com/user-attachments/assets/4403f98e-38df-4b18-9d38-d3793afb9194)


The issue we face is that the number of extras is large making it tedious to analyze one by one and and we might ask ourselves if an extra that is rarely present or always present has an effect on the price of a car.  Below we plot the boxplots of car prices with respect to the presence of each extra or not. 

What we see:
* There are some very rare extras which can have significant effect on the price of a car (for example armored vehicle or air suspension). We did not perform hypothesis tests to ensure that the differences are statistically significant and we conclude visually at this stage. 
* We can see that cars with extras often found in older cars such as cd player or (air conditioning (old) vs automatic air conditioning (newer)) tend to have a lower price
* We see from the extra leasing that there are some cars posted for leasing and that their prices tend to be lower (this might be because the poster mentions the monthly payment value instead of the sale price)
* We see that recent extras such as apple car play, start and stop, automatic parking, led lights have a significantly higher price

![boxplot_extras](https://github.com/user-attachments/assets/869b8479-b682-49af-a6fe-1dfae5613b3b)


#### 3 - Describe of main numerical variables:

On the describe of the main numerical variables below we notice a high presence of outliers which led us to investigate these cases, it turns out that the dataset contains many abnormal ads:

* raw prices of 1: User mention a price of 1 when they want to negotiate directly with the buyer without mentioning the price
* raw prices of over one million euros: these are either mistakes when entering the price or super cars
* 5% of the data has a price below 1800, this is because many users post ads as a car they are selling for spare parts. Multiple ads also concern a specific spare part being sold.
* registration year in 2026 although we are in 2025
* registration years from 1901: these are often antiques or anomalies
* engine powers and sizes of 0 or 1 when the user doesn't know what to write.
* engine sizes below 87cc: these concern some very rare vehicles sometimes with one seat or anomalies
* Cars with 1 seat 

|       |        raw_price |    mileage |   registration_year |   engine_size |   engine_power |         seats |         doors |
|:------|-----------------:|-----------:|--------------------:|--------------:|---------------:|--------------:|--------------:|
| count | 112944           | 112938     |        112944       |    112944     |    112944      | 112908        | 112938        |
| mean  |  14954.4         | 137523     |          2011.54    |      1608.94  |       133.638  |      4.7345   |      4.45693  |
| std   |  19725.4         |  94215.8   |             9.02256 |       661.385 |        83.9379 |      0.899407 |      0.972346 |
| min   |      1           |      0     |          1901       |         0     |         0      |      1        |      2        |
| 1%    |    100           |      0     |          1977       |        87     |         1      |      2        |      2        |
| 5%    |   1800           |    260     |          1997       |       998     |        65      |      2        |      2        |
| 10%   |   3000           |  25000     |          2001       |      1000     |        71      |      4        |      3        |
| 25%   |   6000           |  78000     |          2007       |      1248     |        90      |      5        |      4        |
| 50%   |  10500           | 129000     |          2014       |      1500     |       115      |      5        |      5        |
| 75%   |  17000           | 185000     |          2018       |      1800     |       150      |      5        |      5        |
| 90%   |  28300           | 249000     |          2020       |      2171     |       220      |      5        |      5        |
| 95%   |  41000           | 290000     |          2022       |      2996     |       300      |      5        |      5        |
| 99%   |  90000           | 400000     |          2024       |      4297     |       475.57   |      7        |      5        |
| max   |      1.11111e+06 |      2e+06 |          2026       |     10000     |       999      |     11        |      7        |

We see that we will need to apply some rules based on domain knowledge in order to fix some of these issues as well as maybe try an outlier removal technique in order to remove certain abnormal points. 

#### 4 - Distribution of the target variable:

Here we plot the histogram, box plot of our target variable:

* We can see that when passing to the logarithm the data resembles more a symmetric bell curve. We will not suppose normality without a hypothesis test but what this tells us is that if we wish to remove outliers using a method like z score or interquartile range we should use the logarithm of this feature. 

![price_boxplot](https://github.com/user-attachments/assets/227fbad7-43bf-495e-b139-6d96f5d1d53c)

![image](https://github.com/user-attachments/assets/7fa20ec0-8721-4fa0-8491-678d763ff15f)


#### 5 - Distribution of other numerical variables and comparison with logarithm:

We can see that for the engine power and size there isn't a strong need to pass to the logarithm to suppress outliers. It would be detrimental regarding mileage.


![mileage_dist_log_vs_no_log](https://github.com/user-attachments/assets/280c822d-b468-4e2f-9027-824a3f86c24e)
![engine_power_dist_log_vs_no_log](https://github.com/user-attachments/assets/350e782f-fa2e-44ad-a6a4-18d241ea1bb1)
![engine_size_dist_log_vs_no_log](https://github.com/user-attachments/assets/f4af4dc1-414b-429b-add0-72292771d8a7)
![rim_size_dist_log_vs_no_log](https://github.com/user-attachments/assets/bd7ed51c-ad7f-4ed4-85e9-6bf6d02c0b64)


#### 6 - Correlations and effects between variables: 

If we look at distributions and relationships on the whole dataset we cannot see much. This is logical, we are having high end brands like bentleys and porsches mixed with budget brands such as Dacia or Renault. Therefore a Porsche with 150 000 km mileage might still be more expensive than a brand new Dacia. The same can be thought of within a brand, for Audis for example we will have high end models such the Q8 RS mixed with audi a1s. We can also think about the model year, some brands might release an upgraded car model from a year onwards. It is therefore important in our case to look at relationships within groups. There is a compromise to find however between the level of granularity of the groups. In fact if grouping by brand, model and year we might get groups with not enough points to draw reliable conclusions from.

In fact if we look at the number of points per group at the level brand, model and year we can see that 90% of the groups have less than 24 points within them and we see that we get 11559 groups :

|       |       count |
|:------|------------:|
| count | 11559       |
| mean  |     9.72229 |
| std   |    20.2159  |
| min   |     1       |
| 1%    |     1       |
| 5%    |     1       |
| 10%   |     1       |
| 25%   |     1       |
| 50%   |     3       |
| 75%   |     9       |
| 90%   |    24       |
| 95%   |    43       |
| 99%   |   100.84    |
| max   |   279       |

If we go down a level (brand and model), we have 1621 group and 75% the groups have less than 39 points

|       |     count |
|:------|----------:|
| count | 1621      |
| mean  |   69.3276 |
| std   |  209.384  |
| min   |    1      |
| 1%    |    1      |
| 5%    |    1      |
| 10%   |    1      |
| 25%   |    2      |
| 50%   |    8      |
| 75%   |   39      |
| 90%   |  166      |
| 95%   |  318      |
| 99%   | 1028.2    |
| max   | 2773      |

If we look at a brand level we have 121 brands and 50% of the brands have less than 19 points:

|       |    count |
|:------|---------:|
| count |  121     |
| mean  |  933.421 |
| std   | 2045.22  |
| min   |    1     |
| 1%    |    1     |
| 5%    |    1     |
| 10%   |    1     |
| 25%   |    4     |
| 50%   |   19     |
| 75%   |  409     |
| 90%   | 4426     |
| 95%   | 6660     |
| 99%   | 7723     |
| max   | 9604     |

In fact this shows us that we are dealing with critically important categorical features with high cardinality and a high level of sparsity. We will have some brands which are not present on the Greek market but which still exist and for which we have very few examples. Also, we see that we will have a few select brands under which most of the data is concentrated (Peugot, Fiat, Audis, Bmws, Mercedes etc.). This is something that we will have to keep in mind for the rest of this project. 


Now let's focus on a brand (Mercedes which is the brand with the most data points)



