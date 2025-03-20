# :car: :greece: Machine Learning Based Car Price Estimator

# 🔗 Link to Streamlit App: https://streamlit-app-305336925991.europe-west1.run.app/ 

# 📌 Table of Contents  

- [🤖 How Does This Work?](#-how-does-this-work)  
- [🗺️ Context of This Project](#%EF%B8%8Fcontext-of-this-project)  
- [🎯 Project Goals](#-project-goals)
- [Methodology and Analyses](#methodology-and-analyses)
  - [🕷️ Scraping](#%EF%B8%8F-scraping)  
  - [🧹 Data Cleaning](#-cleaning)  
  - [📊 Exploratory Data Analysis](#-exploratory-analysis)
  - [⚙️ Model Training](#modelling-the-data)  
    - [📊 Baseline Models](#base-model)  
    - [🚀 Advanced ML Models](#catboost)  
    - [🔧 Hyperparameter Tuning](#hyperparameters)  
  - [🎯 Model Performance](#analysis-of-the-chosen-model)  
  - [📉 Error Analysis](#error-analysis)  
  - [🛠️ Model Interpretability](#interpretability)  
    - [🔍 Feature Importance](#feature-importances)  
    - [📉 ICE (Individual Conditional Expectation) Plots](#ice-individual-conditional-expectation-plots)  
    - [📊 Reliability Score Calculation](#reliability-of-the-estimation)  
- [🚀 Deployment](#deployment)  
- [🔍 Challenges, Limitations, and Future Improvements](#conclusion-challenges-weaknesses-and-potential-improvements)  
- [📬 Contact](#contact)  


## 🤖 How does this work?  
This **machine learning model** estimates the price of a used car based on various features such as **brand, model, mileage, registration year, engine, and more**.  

📊 The model was trained on real **used car listings from Greek websites**, so the predictions reflect the **Greek market**.  

**Simply fill in the details** and click the **Predict** button to get:
- An **estimated price** for your car 💰
- A **price range** (low & high) 📈
- A **confidence level** for accuracy 🎯
- An analysis of the **effect of the main features on your car price** 📊


## 🗺️ Context of this project:

Determining the market value of a used car is challenging due to numerous influencing factors. In fact, the price of a used car isn't just a linear function of its mileage and engine size. Other factors such as supply and demand, condition of the car, color, fuel type, interior materials, extras, brand, model, registration year often have a strong impact on a used car's price. Given these complexities, gaining insights from data could help better understand pricing dynamics and help in making more objective estimations of a car's price.  

There are numerous car listing websites on the Greek market and we aim to leverage one of these platforms where sellers list their vehicles characteristics, and buyers can reach out to them.

After a two-year break from data science, this project serves as a hands-on opportunity to refresh and update my skills. The main technical focus areas include:

    Web Scraping & Data Storage – Extracting and storing information from the web via apis or html code 
    Data Preprocessing & Exploratory Analysis – Extracting meaningful insights from real raw data.
    Machine Learning for Price Prediction – Building and evaluating pricing models.
    Implementing a RAG-based Chatbot – Exploring state-of-the-art retrieval-augmented generation techniques. (coming soon)
    Applying deep learning for feature extraction. (coming soon)


## 🎯 Project Goals

To explore these challenges, I aim to: 

      💾 Gather and store all published used car listings  
      🧹 Extract, clean and preprocess the listings to create a dataset representative of today's Greek market 
      📊 Analyze the data to gain insights and a better understanding of the dataset
      ⚙️ Use machine learning to train a car estimation model, comparing multiple preprocessing and modelling techniques and optimising performance
      🚀 Deploy the model using a containerized Streamlit app on Google Cloud Run

Next steps: 

      🤖 Chatbot using Retrieval-Augmented Generation (RAG): Enable users to query listings using natural language.
      🔍 Text-Based Behicle Condition Assessment: Fine tune a large language model for sentiment analysis on car listing descriptions
      🖼 Image-Based Feature extraction: Extract Vehicle features from car listing images by fine tuning computer vision transformers to enrich missing fields.
  

## Methodology and Analyses:

### 💾 Scraping

The first step consisted in scraping car listing from the web through identified apis. When looking at car listing in web developer mode in my browser, the displayed content of the listings was extracted from the response to an api request in the form of a large json file of around 3000 rows with various information about the listing. We can find links to the pictures, informations about the listing, dates of publishing and modifications, extras of the car, main characteristics and many other fields. Below is a screenshot of a small section of such a record:

![image](https://github.com/user-attachments/assets/8ca75dc0-ac6d-4b4d-ae62-f667a13fcbb3)

There are many fields which aren't relevant to our use case, so the first step was to list the keys of the json which interested us. 

Each car listing has an id and we wish to extract all the car listings. So the next step was to find all the ids on the website. To do this, we use a search on the website without any filter which leads to a paginated list of all ads. This information was also the result of an api request. For each page of size n_listings, an api request is made to the page number and a json with the n_listings listed on the page are returned. What was useful is that these jsons were much smaller and contained the main characteristics of the listed cars with their ids.

So the methodology to scrape the data was the following: 

* Create an empty dictionnary
* For each page send an api request to get the n_listing listings
* For each listing in the page fill the dictionnary with id as the key and the json with listing information
* Save the final dictionnary as a dataframe

At this stage we get a dataframe with columns containing lists of mixed types or lists of dictionaries (see image below):

![image](https://github.com/user-attachments/assets/c9024264-d3c8-49e3-b0cc-13e3abbdb62e)

We see that the fields are formatted in various ways, some are grouped under dictionnaries with multiple keys, the price is stored as a string with a space between thousands and the EUR sign, mileage is a string with KM sign. That is what we will come to treat in the next section (Cleaning). 

Before this however, we then used all the collected ids to scrape the ads one by one to obtain car characterstics which weren't present in the crawling on listing pages such as extras, interior types etc. This allowed us to enrich the data with additional features for each car ad. 

### 🧹 Cleaning

This data was messy and inconsistent considering the way it was stored and the types (no standard json type for each field within the api response, meaning that each field had to be treated with a custom strategy. Presence of mixed types within various fields, floats stored as strings with both numbers and letters and special characters). The largest part of the project was spent on this stage and below are the various cleaning methods that we applied to this data.  


✔️ Extracted all possible extras in order to convert extras from a list of jsons each representing the mention of an extra to one boolean column per extra (example: air_conditioning: True or False)  
✔️ Extracted all possible specifications to convert specifications from one json with mentioned specifications to one column per specification containing their respective values (example fuel_type: petrol)  
✔️ Extracted all possible trim levels to convert model trim column from a json with mentioned trim specs to one column per trim spec (example one column doors: number of doors) for each listing  
✔️ Extract description and store it in one column for NLP applications
✔️ Deduplicated ads, some listings which were listed with paid priority would appear multiple times on the website so we had to keep only one listing per id  
✔️ Droppped over 40 unnecessary columns (leasing, marketplace, seo, finance options etc.)   
✔️ Processed geolocation json to extract latitude longitude  
✔️ Converted mileage from a string of the form 15 000 KM to float using regex  
✔️ Extracted battery charge time, fuel type, engine power, gearbox type, features, engine size, battery range from key feature dictionnaries  
✔️ Extracted Brand, model, variant, registration year from dictionnary  
✔️ Converted engine size from 1200 cc/375kW/1000 W to float  
✔️ Converted dates to datetime  
✔️ Deduplicated redudant columns (some columns were redundant as extracted from different sections of the json)  
✔️ Removed columns with constant values for all listings  
✔️ Converted battery range from 400 km or 400 χλμ to a float via regex  
✔️ Converted battery charge time from 7 ωρες το an integer  
✔️ Translated from Greek to English, Mapped and Merged redundant category levels for multiple features (fuel type, gearbox_type, interior type, exterior color, interior color etc.). We encountered multiple features such as fuel type where the levels were stored both in English or in Greek within the same column. For example we would have some cars with fuel type Πετρελαιο and some with Diesel. We merged each level together for these columns.   
✔️ Converted year column stored as double digit string in the form 00 for 2000 and 92 for 1992 to the 4 digit year  
✔️ Extracted the information Μεταλλικο regarding exterior color to store it as a boolean feature is_metallic  
✔️ Converted emissions Co2 from 98 g/km to float  
✔️ Converted rim size from 17 inches or 17 ιντσες to integer.  
✔️ Filled missing co2 emission values for electric cars to 0  
✔️ Merged inconsistent category levels of body type (Bus with Van and van ) which all consered mini vans  
✔️ Dropped listings without a price  

### 📊 Exploratory Analysis 

#### Dimensions:

We obtain a tabular dataframe with  112944 listings and 145 features

#### Missing values: 

We notice some features with very low level of completion <50% such as vehicle dimensions, performance features such as torque, technical check up, battery related information and co2 emissions. Also, we see the extra features all with the same level of completion, this is because when the json containing extras was present we set to true the extras contained in it and to false the ones not mentionned. Therefore all extras are set to NaN for listings which did not contain a json with extras.

![completude](https://github.com/user-attachments/assets/1682b5b6-d858-4eee-a2a9-d1e1a45549c2)

#### Analysis of fields extracted from extras: 

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


#### Describe of main numerical variables:

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

#### Distribution of the target variable:

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

#### 6 - Correlations

Below we plot the correlation matrix between the main numerical features in our dataframe. We observe:

* Positive correlations between price and: engine power, engine size and registration year
* Negative correlations between the price and the mileage
* An obviously high correlation between engine power and engine size.
* Almost no correlation between price and number of doors and seats. This is strange as we might intuitively think that the higher number of seats, the larger the vehicle and therefore the higher the price.
* A high correlation between number of doors and number of seats

![correlation](https://github.com/user-attachments/assets/355070d5-80f8-457f-82e1-81e255eee47d)

Here we plot the correlations between the price and the presence of an extra in a car, we observe multiple extras with a null correlation such as exchange with bike which even intuitively don't seem impactful on the price. We also see some positivetly correlated extras which are either premium features( panormaic roof, heated seats, air suspension) or extras which are correlated with the registration year (apple car play, parktronic, automatic parking, eco start stop, lane assist) which are representative of newer models. As a result it will be useful to remove extras with low correlations or not directly impacting the price of a car. 

![extra_corr](https://github.com/user-attachments/assets/850fdb73-086f-4913-8959-0d620e999730)



#### 7 -  Relationships between variables: 

If we look at distributions and relationships on the whole dataset we cannot see much. This is logical, we are having high end brands like bentleys and porsches mixed with budget brands such as Dacia or Renault. Therefore a Porsche with 150 000 km mileage might still be more expensive than a brand new Dacia. The same can be thought of within a brand, for Audis for example we will have high end models such the Q8 RS mixed with audi a1s. We can also think about the model year, some brands might release an upgraded car model from a year onwards. As we can see this is clear when looking at the median prices per brand, model and year:

![median_price_per_brand](https://github.com/user-attachments/assets/5084f118-f344-4032-8e86-a0b410f23d58)
![median_price_per_audi_model](https://github.com/user-attachments/assets/a2271a85-4494-44af-be10-5117f355f5eb)
![boxplot_per_year_price](https://github.com/user-attachments/assets/9975cf6a-8c0c-4dd7-812f-c0ba4e06ad0e)


It is therefore important in our case to look at relationships within groups to gain valuable insights regarding the relationships between certain features, but also to clear outliers! We wouldn't want to clear outliers on the price on the whole dataset as this would remove high end cars and budget cars only. They have to be considered by group.

There is a compromise to find however between the level of granularity of the groups. In fact when grouping by brand, model and year we might get groups with not enough points to draw reliable conclusions from.
If we look at the number of points per group at the level brand, model and year we can see that 90% of the groups have less than 24 points within them and we see that we get 11559 groups. If we go down a level (brand and model), we have 1621 group and 75% the groups have less than 39 points. If we look at a brand level we have 121 brands and 50% of the brands have less than 19 points:

|       |   brand_model_year_group_count |   brand_model_group_count |   brand_count |
|:------|-------------------------------:|--------------------------:|--------------:|
| count |                    11559       |                 1621      |       121     |
| mean  |                        9.72229 |                   69.3276 |       933.421 |
| std   |                       20.2159  |                  209.384  |      2045.22  |
| min   |                        1       |                    1      |         1     |
| 1%    |                        1       |                    1      |         1     |
| 5%    |                        1       |                    1      |         1     |
| 10%   |                        1       |                    1      |         1     |
| 25%   |                        1       |                    2      |         4     |
| 50%   |                        3       |                    8      |        19     |
| 75%   |                        9       |                   39      |       409     |
| 90%   |                       24       |                  166      |      4426     |
| 95%   |                       43       |                  318      |      6660     |
| 99%   |                      100.84    |                 1028.2    |      7723     |
| max   |                      279       |                 2773      |      9604     |


We understand that we are dealing with critically important categorical features with high cardinality and a high level of sparsity. There are some brands which are not present on the Greek market and for which we have very few examples. On the other hand, we have a few select brands under which most of the data is concentrated (Peugot, Fiat, Audis, Bmws, Mercedes etc.) This is something that we will have to keep in mind for the rest of this project. 

![pct_listings_per_brand](https://github.com/user-attachments/assets/1cec9444-b65c-4c1b-984d-f393afc7482e)  

Now we showcase this by choosing one brand: BMW which is one of the most present brands in the dataset and which offers a wide range of models from budget to high end models. We only visualise three models (X1, X5, M8) one lower end model, one mid range model and one high end model. We only keep cars from 2015 and later. We can clearly see three separate distributions for each model if we look at the log of the raw price with respect to other features. And we see that for each model there is a linear relationship between the log of the raw price and the mileage, between the log of the price and the registration year. 

We also see at the brand level, a linear relationship between  the log of the price and the engine power and engine size.

![pairplot_for_bmw_and_3_model](https://github.com/user-attachments/assets/307a0d20-d5b0-476c-be58-0305fd4d0f66)  

If we plot the pairplot by fixing the model to the bmw 116 and look at three distant year models 2005, 2012 and 2020, we also see 3 distinct price distributions. For the same mileage we see that the newer model the higher the price is. This reinforces the importance of considering groups of listings on these features when we wish to process data or to fit certain models. Again if we suppressed outliers without grouping, we could end up suppressing only old cars and new cars.

![pairplot_for_bmw_and_3_years](https://github.com/user-attachments/assets/d2df3b71-7af1-40ef-a05d-4a7089286116)


### Processing

Now thout we gained some insights on our data and identified abnormalities we can process the data further. The following processing was applied to the data:

* Remove missing model names
* Only keep registration years above 2000 (this was chosen based on domain knowledge and based on what we want to achieve i.e: estimate the price of common used cars on the market)
* Remove registration years equal to 2026
* Remove antiques
* Remove prices below 500 euros: we consider these points abnormal based on domain knowledge and if needed will apply further outlier removal strategies later. We also saw some car listings with prices of 50 meaning 50 000 in the data so this helps remove these cases
* Remove prices above 350 000 euros
* Remove listings with leasing set to true
* If there are listings with crashed set to  True and never_crashed set to True this is a contradiction so we correct the values of never crashed to False
* We removed extras which are always set to False or with a percentage of Trues lower than 3.5% (except for armored car and wheelchair which we judge informative in terms of domain knowledge)
* We removed other extras displaying low correlations and which don't seem informative such as 'extra_greek_dealership','extra_imported','extra_exchange_with_bike','extra_credit_card_accepted','extra_acc','extra_nonsmoker' etc.
* We removed listings with nans on doors and seats as they represented less than 10 listings.

Now considering the large number of extras, my question was if there was a way to simplify the dataset and reduce the dimension of the feature space. I grouped the extras in 6 categories:  

1 - Comfort
2 - Infotainement connecitivty
3 - Safety and Driver Assistance 
4 - Performance 
5 - Utility (wheelchair, hitch, service book)
6 - Premium (armored, air suspension etc)

Then for each listing $x$ I created a simple option score $S_o$ between 0 and 1 for each category equal to the percentage of True values within the category:

$S_o (x) = \frac{1}{d_{c}}\sum_{i=1}^{d_{c}} \mathbb{1}_{x_i=1}(x_i)$ where  $d_c$  is the number of extras within the category

We created these columns to try a model using only these 6 columns instead of the extras. 

The advantage of this method is that we simplify the model through dimension reduction while keeping interpretability and as we see on the correlation plot below the categories positively correlate with the price. 

The disadvantages of this method is that we loose interpretability about the impact of an individual extra, for example a listing with only apple car play set to True in infotainment will have the same score as a car with only cd player and therefore we degrade our ability to interpret how a specific extra might affect the prediction, to tackle this we could have weighted each extra in the score with the normalised correlation coefficients with respect to the price for example (this could be a future experiment). Another disadvantage we see on the correlation plot below is that these new are correlated and therefore would not be interesting in models such as a linear regression. Since we plan to use ensemble methods, correlated features aren't as detrimental and we will therefore try using these. 

![image](https://github.com/user-attachments/assets/4f3038a4-cec8-4d57-b8d4-920969c28978)


### ⚙️ Modelling the data:

Through this project we focused on three main modelling techniques:

* A **base model** consisting in grouping the training data and estimating the median price per group
* An improvement on the base model, fitting **linear regression on groups** within the training set and using the corresponding fitted regression to predict the value of a car given its group
* A **gradient boosting** model, catboost which is ideal for tabular data with categorical features with high carinality and missing values

We noticed significatively higher performance from catboost and tried multiple different preprocessing techniques in order to find the most accurate method.

The data was split in a training and test set with an 80/20 ratio. 3 Fold cross validation was used to train and validate catboost models. 

We display the main results below and then provide a description of each method. 

We see on the table below that our chosen model (**caboost drop unpractical**) achieves a mean absolute percentage error of **14.7%** and a median absolute percentage error of **8.9%**.

|    | strategy                    |     MAPE |    MedAPE |     MAE |    MedAE |
|---:|:----------------------------|---------:|----------:|--------:|---------:|
|  0 | catboost all extras         | 0.146884 | 0.0873134 | 1950.77 |  926.606 |
|  1 | catboost no extras options  | 0.149857 | 0.0888384 | 1971.19 |  950.514 |
|  2 | catboost drop low importance| 0.147278 | 0.0876012 | 1928.97 |  934.133 |
|  3 | catboost drop unpractical   | 0.1477   | 0.0890763 | 1964.05 |  948.112 |
|  4 | catboost impute missing     | 0.147684 | 0.086413  | 1925.41 |  925.372 |
|  5 | catboost options            | 0.147313 | 0.0878808 | 1951.26 |  926.826 |
|  6 | catboost remove outliers    | 0.153436 | 0.0886593 | 2037.55 |  930.431 |
|  7 | linear regression           | 0.191793 | 0.122594  | 2682.94 | 1284.06  |
|  8 | base model median           | 0.220556 | 0.122727  | 2740.67 | 1300     |



We previously saw that the data contained multiple outliers even after cleaning obvious abnormalities. If we restrict ourselves to more reasonable points, that is prices between 2000 and 100 000 euros, model years above 2008, and engine sizes above 600cc we observe a mean absolute percentage error of 10%, median absolute percentage error of 7.1% and errors below 21.27% 90% of the time as shown on the table below.


|       |   ape_drop_unpractical |
|:------|-----------------------:|
| count |        14368           |
| mean  |            0.100533    |
| std   |            0.158236    |
| min   |            1.24249e-05 |
| 1%    |            0.00107466  |
| 5%    |            0.00564499  |
| 25%   |            0.0313831   |
| 50%   |            0.0717674   |
| 75%   |            0.131842    |
| 90%   |            0.21271     |
| 95%   |            0.277772    |
| 99%   |            0.480234    |
| max   |           13.2074      |

We chose the **drop unpractical** method as it made it easier for the end user of the estimating tool and as it did not imply to significantly sacrfice performance with respect to the other methods.  

#### Base model:

We first try a simple estimation method, grouping cars in three granularity levels:

* By brand, model , year
* By brand, model
* By brand, year
* By brand

For each car to estimate in the test set we find its corresponding most granular group in the training set first (by brand, model, year). If there are enough points in the group (greater than a fixed threshold) we estimate the price as the median price of the corresponding group in the training set. If the number of points in the group is lower than the set threshold, we move to a group level with less granularity.

The value of the threshold is a hyperparameter here. For a value of 1, the model will fit the training data to the finest level in every case. The higher the threshold, the harder it will be for a car to be estimated based on the finest level and therefore we risk underfitting. Below we will plot the mean absolute percentage error on the test set with respect to different thresholds. 

![group_median_thresh](https://github.com/user-attachments/assets/b39cdbc9-a64d-45e3-a32e-4ed7b197b050)


With a threshold of 1 we obtain a mean absolute percentage error of 22%. Below is a describe of the residuals, absolute residuals and absolute percentage errors. We see that in median the absolute percentage error is of 12.2%

|       |   residuals_median |   abs_residuals_median |     ape_median |
|:------|-------------------:|-----------------------:|---------------:|
| count |          20286     |               20286    | 20286          |
| mean  |            537.334 |                2740.67 |     0.220556   |
| std   |           6822.58  |                6270.94 |     0.585012   |
| min   |        -120000     |                   0    |     0          |
| 1%    |         -12499.1   |                   0    |     0          |
| 5%    |          -4545     |                  99    |     0.00742092 |
| 10%   |          -2810     |                 200    |     0.02       |
| 25%   |          -1200     |                 550    |     0.05375    |
| 50%   |              0     |                1300    |     0.122727   |
| 75%   |           1480     |                2700    |     0.237179   |
| 90%   |           3990     |                5600    |     0.421623   |
| 95%   |           6750     |                9137.75 |     0.615908   |
| 99%   |          20500     |               24601    |     1.99615    |
| max   |         203074     |              203074    |    51.0301     |

✅ Advantages of this model: 
 
  *simple approach
  *easily interpretable

❌ Disadvantages of this model: 

* Important charactersitics of the cars aren't taken into account (engine specs within a model, extras, gearbox type)
* Very dependent on the representation of car models in the data
* Problematic when a brand has a wide range of models from budget options to high end options (in the case where we have to estimate the price at the brand level for example)

  
When looking at examples  with the largest errors below, we make errors on cars with high mileage since this is not taken into account. We also notice large errors due to outliers, judging by the descriptions, some cars are posted for leasing with the mentionned price being the monthly dose. We see cars being sold for spare parts and an ad where the user sells only the engine with 250 000 km and mentions the mileage as 0. 

![image](https://github.com/user-attachments/assets/307ce0b7-296d-4a7e-878d-9668c485bd80)

Below we plot the mean absolute percentage error on the test set based on the group threshold and we see that the finer the group, the better the estimation:

![image](https://github.com/user-attachments/assets/ac9b4fda-8597-44d4-a6aa-f29dd48be6a8)

We also see that the model peforms badly on crashed cars, this is logical considering our approach.

![image](https://github.com/user-attachments/assets/6cd0b616-f376-4b15-a8f7-16ab74a5c71d)



#### Grouped Linear Regression:

We then try a similar approach with linear regression. 
  
We observed during the exploratory analysis linear relationships between the price of a car (log transformed) and certain features. So linear regression seems like an adequate method to explore in order to model our data.   
  
However, we also saw that these linearities appeared on groups and will therefore have to group the data again. Since we saw a linear relationship with the registration year we will use this feature as one of the independent variables. That is why in this method we group the listings in the following way:  
  
* By brand and model  
* By brand
  
We model the price of the car as a linear regression with respect to mileage, engine_size, registration_year, crashed. We did not include the engine power as we saw that engine power and size are highly correlated, something which goes against linear regression assumptions. We did not include extras, interior or exterior colors and types as these variables had a large percentage of missing values.
  
Let's take a look at the fitted linear regression on the group Citroen C3:  
  
We see a well conditioned covariance matrix, statistically significant estimated coefficients with respect to the p value for each coefficient. We see that the most impactful variable is the registration year with a positive effect on the price, then we see that the mileage and the is_crashed variables affect the price negatively and finally the engine size which also impacts the price positively. We see an adjusted R2 of 0.89 meaning that our model captures most of the variance in the model.  
  
In this approach we scaled the features as otherwise we encountered ill conditioned covariance matrices leading to instabilities. We also took the log transformation of the price for fitting the regressions as we saw that the linear relationships were stronger when looking at the logarithm.  
    
![image](https://github.com/user-attachments/assets/1bf22ac4-6cc1-4043-bf6e-972038974bce)
   
We split the training set in order to get a validation set and experiment with thresholds to see which one minimises the error. We estimate that 15 is the ideal threshold: 
  
![threshold_linreg](https://github.com/user-attachments/assets/645dff8b-c61a-41c3-9f79-063ce474fb58)


After refitting the regressions on the whole training set these are the results we obtain: 

|       |      ape_linreg |
|:------|----------------:|
| count | 20286           |
| mean  |     0.191793    |
| std   |     0.513715    |
| min   |     1.89398e-05 |
| 1%    |     0.00207039  |
| 10%   |     0.0218623   |
| 50%   |     0.122594    |
| 90%   |     0.383605    |
| 95%   |     0.530776    |
| 99%   |     1.14574     |
| max   |    51.6949      |

We had to use a threshold of 20 however as instabilities were encountered using 15 on the whole training set. We observe an improvement with respect to the base model both in mean and median. 

We notice better estimations when the regressions are able to be fitted at the model level rather than at the brand stage:

![image](https://github.com/user-attachments/assets/e47b2245-90b8-46d5-9844-eb7eb639e867)

We notice that the method estimates newer models more precisely than older models:

![image](https://github.com/user-attachments/assets/1f76e35e-c2bc-4861-a914-2ea688c52d99)

Residuals seem to be normally distributed

![image](https://github.com/user-attachments/assets/c883fcaf-5499-4b15-a856-812706df399f)

✅ **Advantages of this model: **
 
  * simple approach
  * easily interpretable
  * Leveraging linear relationships within the data

❌ **Disadvantages of this model: **

* We couldn't include all our categorical variables as this would lead to a very high number of one hot encoded features
* Imputing missing values on very sparse features is necessary and could be misleading
* Other features such as co2 emissions or vehicle dimensions aren't necessesarily linear with the price
* We rely on groups again, and less represented car models and years will not be estimated well
* We store over 1000 models. Unpractical to deploy in production. For each car we wish to estimate, we might need to load a different model 


#### Gradient Boosting (Catboost)

As we saw with the base model and the linear regression we were forced to fit models on groups of cars using model and brand and sometimes registration year. The problem encountered with these methods were that for groups with very few observations we could not fit a model on the desired level of group granularity and therefore had to fit the model on a broader group (for example a porsche 918 would have to be estimated using a regression fitted on all porsches since there weren't enough of these cars in the dataset). This lead to worse estimations for car models which are not present enough in the dataset.  
  
Moreover, with linear regressions we were limited to using variables which displayed linear relationships with the price, however not all independent variables in our dataset have a linear relationship with the price (for example the latitude and the longitude).  
  
Ensemble methods such as Random Forests and Gradient boosting, based on fitting regression trees are still today the leading methods for tabular datasets. The advantages of these methods are their ability to capture complex relationships, handle categorical variables efficiently while remaining interpretable (although gradient boosted trees aren't as interpretable as compared to random forests). These methods by nature split the data into groups and are more robust to outliers.  
  
Gradient boosting consists in sequentially fitting multiple shallow trees (base learners) each fitted tree is fitted on the errors of the previous one.  
  
In our case gradient boosting and specifically Catboost stands out as an ideal method. In fact Catboost is a gradient boosting algorithm which is tailored to datasets with sparse categorical features with high cardinality which it treats via an advanced version of target encoding (consisting of assigning the mean value of the target variable for each category level). It also handles missing values automatically, treating them as a separate category. Moreover Catboost support training on GPU reducing significatively the training time which is important since such models require tuning hyperparameters on large grids.  
  
Some of our categorical variables might intuitively have a great influence on the price of a car (for example the extras) however, simple imputation methods for these variables would affect the structure of the data. In fact for the extras the NaN value means that we do not know if the car has this extra or not since the users did not bother to fill the extras fields, and therefore these should be treated as a separate category.  
  
The main features we have at our disposal on top of the the extras and the engineered options columns we discussed before were the following:  
  
'lat', 'lon', 'is_new', 'mileage', 'crashed', 'engine_size', 'registration_month', 'registration_year', 'engine_power', 'fuel_type', 'gearbox_type', 'brand', 'model', 'interior_type', 'seats', 'kteo','exterior_color', 'number_plate_ending', 'emissions_co2', 'battery_charge_time', 'interior_color', 'rim_size', 'vehicle_height','number_of_gears', 'torque', 'gross_weight', 'acceleration', 'vehicle_width', 'body_type', 'vehicle_length', 'top_speed','wheelbase', 'fuel_consumption','drive_type', 'doors', 'is_metallic'
  
We gained signficant improvements in performance using this method and tried multiple preprocessing methods in order to compare performances (described below).  
  
In order to compare methods we used 3 fold cross validation on the training set and optimised hyperparameters using optuna. Optuna is a hyperparameter optimisation tool library leveraging tools such as Bayesian optimisation and given a grid of hyperparameters it iteratively learns the optimal distribution of hyperparameters while pruning unpromising trials before the end of a run. We set the number of runs to 30 for each method. We also sixed a large number of iterations (15000) and set the overfitting detector in order to stop training when the validation loss wouldn't decrease for over 50 iterations. 
  
We choose to optimise the MAE loss (same as estimating the median price instead of the expected price). This choice was made because our dataset contains multiple outliers.  
  
**1 - All Extras model**: This method consisted of training catboost on all the above features, only dropping the options constructed columns. We kept missing values as nans, and all text categorical features as strings as Catboost preprocesses them automatically  
  
**2 - Options model**: This method consists of keeping all features above but dropping the boolean extras columns. The extras are therefore taken into account through the constructed options columns discussed before 

**3 - Outlier removal**:  We constructed a method to remove outliers. In order to do this we clean outliers on the **price**, the **engine size**  and the **engine power** using the **interquartile range method**. We only remove outliers on the training set and evaluate the performance on the whole test set. For the **price** we removed outliers on the **log transformed price** as the price itself wasn't symmetric and we detect abnormal prices on groups and thresholds again. We first group at the most granular levels, for groups with over 40 points, we compute the bounds of the interquartile range and remove values outside of it (beyond 1.5 times above and below bounds).  For the groups with less than 40 points we regroup them at a less granular level (brand and model) and compute the IQR on the groups. Then for the remaining groups with less than 40 points we use the brand level. and for the remaining groups we compute bounds on the whole set. For the engine size and power we use IQR on the whole set without grouping. For each row we evaluate if it is an outlier with respect to each of the three features. If a row is an outlier with respect to at least one of them we delete it. This represented 6% of the data approximately.   

**4 - Drop unpractical**: Keeping a product oriented mindset, we drop features which make the estimation tool too complicated to use for the end user. In fact, it would be very tedious for each estimation of a car to know the acceleration, torque, vehicle dimensions, weight, number of gears, co2 emissions. So we try to restrict the features to the extras and to the main characteristics which a user is expected to know. Also, we know that the dropped features are highly correlated to  the car model, year and engine size in general. And when looking at the feature importances of catboost on all features we saw that these very specific characteristics were scoring low. 

**5 - Drop low importance:** We fit a catboost model by dropping all feature importances of the all extras model lower than a certain threshold.  

**6 - Impute missing values:** In this method we try to impute missing values in order to see if this could improve performance using the all extras dataset. The imputing strategy was again based on groups. We compute the mean value on groups for numerical values and the mode for categorical features (majority vote). If the group only has missing values within a group, we group at a less granular level.  


✅ **Advantages of this method: **

* Categorical variables handled automatically and intelligently
* Robust to outliers
* No need to group data
* Performance isn't affected by multicolinearity

❌ **Disadvantages of this method: **

* Less interpretable: cannot trace how the prediction was made although can understand which features have the most impact
* Heavy hyperparameter tuning
* Long training times
* Heavier model to load

  
**The result comparison of all methods can be found below: **

|       |   ape_all_extras |   ape_drop_extras_and_options |   ape_drop_low_importance |   ape_drop_unpractical |   ape_impute_missing_values |   ape_options |   ape_remove_outliers |   ape_linreg |   ape_median |
|:------|-----------------:|------------------------------:|--------------------------:|-----------------------:|----------------------------:|--------------:|----------------------:|-------------:|-------------:|
| count |        20286     |                     20286     |                 20286     |              20286     |                   20286     |     20286     |             20286     |    20286     |    20286     |
| mean  |            0.147 |                         0.15  |                     0.147 |                  0.148 |                       0.148 |         0.147 |                 0.153 |        0.192 |        0.221 |
| std   |            0.439 |                         0.453 |                     0.445 |                  0.436 |                       0.458 |         0.425 |                 0.45  |        0.514 |        0.585 |
| min   |            0     |                         0     |                     0     |                  0     |                       0     |         0     |                 0     |        0     |        0     |
| 1%    |            0.001 |                         0.001 |                     0.001 |                  0.001 |                       0.001 |         0.001 |                 0.001 |        0.002 |        0     |
| 5%    |            0.007 |                         0.007 |                     0.007 |                  0.007 |                       0.007 |         0.007 |                 0.007 |        0.011 |        0.007 |
| 25%   |            0.038 |                         0.039 |                     0.038 |                  0.038 |                       0.038 |         0.038 |                 0.039 |        0.056 |        0.054 |
| 50%   |            0.087 |                         0.089 |                     0.088 |                  0.089 |                       0.086 |         0.088 |                 0.089 |        0.123 |        0.123 |
| 75%   |            0.17  |                         0.173 |                     0.171 |                  0.172 |                       0.17  |         0.172 |                 0.175 |        0.224 |        0.237 |
| 90%   |            0.302 |                         0.302 |                     0.3   |                  0.304 |                       0.297 |         0.298 |                 0.313 |        0.384 |        0.422 |
| 95%   |            0.423 |                         0.431 |                     0.428 |                  0.425 |                       0.425 |         0.425 |                 0.451 |        0.531 |        0.616 |
| 99%   |            0.967 |                         0.995 |                     0.972 |                  0.944 |                       0.971 |         0.978 |                 1.1   |        1.146 |        1.996 |
| max   |           45.584 |                        47.797 |                    47.282 |                 47.185 |                      49.541 |        43.087 |                47.003 |       51.695 |       51.03  |

The final chosen model is the drop unpractial method which allows us to simplify the user experience of our estimation tool without compromising on performance. It also achieves an adjusted R squared of 0.92. The outlier method seems to perform worse than the other methods but this is logical considering that the model was trained on cleaner data but evaluated on the raw test set. However when we looked at the performance of this method only on non outlier points we did not notice a performance improvement, that is why we did not choose this method. COnsidering that the impute missing values method performs as well as without imputing them we decided not to perturb our data. 

Each of these catboost model was optimised for hyperparamets using 3 fold validation before training on the whole training set, so we are certain that hyperparameters do not influence a method more than another. 

We also tried to impose monotonic constraints on extras and other features but encountered severe performance degradation as well as unstabilities. 

### Analysis of the chosen model 

#### Hyperparameters

The optimal hyperparameters of our model were the folowing: 

{'random_strength': 0.01199857498,
 'verbose': 50,
 'iterations': 11333,
 'nan_mode': 'Max',
 'bagging_temperature': 0.8319957516,
 'grow_policy': 'Depthwise',
 'l2_leaf_reg': 0.6747582331,
 'loss_function': 'Quantile:alpha=0.5',
 'task_type': 'GPU',
 'depth': 9,
 'min_data_in_leaf': 92,
 'learning_rate': 0.02721930604}


#### Feature importance

![feature_importances_catboost](https://github.com/user-attachments/assets/a71b0f92-8bc4-4e67-920b-f3ccd7c87bba)


#### Residuals vs price

![pe_practical_price](https://github.com/user-attachments/assets/71414630-5199-4e9d-92ba-51b2012ef4e0)

#### Performance by registration year

![ape_practical_box_year](https://github.com/user-attachments/assets/3fcd36f6-a1fd-4e6f-9e30-0e46c1d329b9)


#### Residuals vs Engine power

![scatter_engine_power_ape](https://github.com/user-attachments/assets/5b9b90ae-4e27-4844-9758-6b08c2827342)

#### Error vs Brand

![ape_practical_bar_brand](https://github.com/user-attachments/assets/1b2fe732-44b6-4fc9-a7e0-4ecd10845ff5)


#### Error vs fuel type

![ape_practical_box_fuel](https://github.com/user-attachments/assets/f52d91a2-5611-4d4f-8dc1-cc6976f9cb7b)

#### Error if crashed:

![ape_practical_box_crashed](https://github.com/user-attachments/assets/60b198da-376f-438e-8f70-58038713652a)

#### Error vs number of points per group model and year: 

![scatter_group_count](https://github.com/user-attachments/assets/105ba205-5bd4-4cc3-b1b8-7f11ea5c66bf)


Describe of errors with extras missing: 

|       |   ape_drop_unpractical |   ape_drop_unpractical |
|:------|-----------------------:|-----------------------:|
| count |               3834     |              16452     |
| mean  |                  0.21  |                  0.133 |
| std   |                  0.396 |                  0.444 |
| min   |                  0     |                  0     |
| 1%    |                  0.002 |                  0.001 |
| 5%    |                  0.01  |                  0.007 |
| 25%   |                  0.054 |                  0.036 |
| 50%   |                  0.123 |                  0.083 |
| 75%   |                  0.239 |                  0.159 |
| 90%   |                  0.429 |                  0.275 |
| 95%   |                  0.617 |                  0.375 |
| 99%   |                  1.464 |                  0.794 |
| max   |                 13.207 |                 47.185 |


Describe of errors in normal values:

|       |   ape_drop_unpractical |
|:------|-----------------------:|
| count |        13864           |
| mean  |            0.099785    |
| std   |            0.155323    |
| min   |            1.24249e-05 |
| 1%    |            0.00109708  |
| 5%    |            0.00592297  |
| 25%   |            0.0318868   |
| 50%   |            0.0722225   |
| 75%   |            0.131778    |
| 90%   |            0.211244    |
| 95%   |            0.274392    |
| 99%   |            0.46893     |
| max   |           13.2074      |
Describe of features for errors > 98th percentile

|       |   raw_price |   registration_year |   engine_size |   mileage |   ape_drop_unpractical |
|:------|------------:|--------------------:|--------------:|----------:|-----------------------:|
| count |      406    |           406       |       406     |       406 |             406        |
| mean  |     7352.75 |          2006.3     |      1934.07  |    168374 |               1.44206  |
| std   |    19620.9  |             4.63368 |       977.081 |    140922 |               2.67159  |
| min   |      600    |          2001       |         0     |         0 |               0.679464 |
| 1%    |      600    |          2001       |         1     |         0 |               0.681008 |
| 5%    |      700    |          2001       |       824.5   |         1 |               0.688687 |
| 25%   |     1500    |          2003       |      1400     |     80000 |               0.785516 |
| 50%   |     2500    |          2005       |      1800     |    157500 |               0.944885 |
| 75%   |     4500    |          2008       |      2107.25  |    222000 |               1.37181  |
| 90%   |     9700    |          2012       |      2998     |    300000 |               2.07766  |
| 95%   |    25000    |          2016.75    |      3775     |    435500 |               2.87455  |
| 99%   |   111111    |          2021.95    |      5275.5   |    671250 |               7.21141  |
| max   |   180000    |          2025       |     10000     |    999999 |              47.1849   |

We see that 90% of the very large errors are on models older  than 2012  and on very low real prices. We also see the case of wrong engine sizes (0 or 1) and mileages set to 0. In fact we saw we had certain points where user enters mileage 0 or 1 for spare parts. 


#### Examples of high errors

We see below that large errors are on outliers with low prices. We see a car being sold as a seasonal rental, cars sold for spare parts, some needing repairs. Of course our model will overestimate the value of such cars and this is where nlp could be useful to improve the performance of our model. In fact appart from is crashed we do not have any information about the condition of the car. 

|       | brand         | model        |   registration_year |   engine_size | description                                        |   raw_price |   pred_unpractical |   ape_drop_unpractical |
|------:|:--------------|:-------------|--------------------:|--------------:|:---------------------------------------------------|------------:|-------------------:|-----------------------:|
| 95705 | mitsubishi    | pajero pinin |                2004 |          2000 | πωλειται το μοτερ απο αυτο το αυτοκινητο.ειναι σε  |        1200 |            4696.73 |                2.91395 |
| 49505 | volkswagen    | polo         |                2001 |          1400 | θελει μαζεματα.κτεο και σημα του 2025              |         650 |            2441.63 |                2.75636 |
| 43440 | skoda         | octavia      |                2012 |          1600 | σε αριστη κατασταση με ολα του τα service δεκτος κ |        2000 |            7501.5  |                2.75075 |
| 62962 | mercedes-benz | vito         |                2025 |          1950 | αφορα μισθωση σεζον,,,καλεστε 6937442443 για περισ |       16000 |           58125.4  |                2.63284 |
| 89302 | mercedes-benz | cla 180      |                2010 |          1600 | αγοραζονται αμεσα τοις μετρητοις τρακαρισμενα   η  |        5000 |           17946.3  |                2.58926 |
| 39274 | opel          | vectra       |                2007 |          1800 | διαφορα ανταλλακτικα απο το εικονιζομενο..μοτερ αψ |         600 |            2140.19 |                2.56699 |
| 16758 | opel          | vectra       |                2002 |             1 | , μονο για ανταλλακτικα παρακαλω                   |         750 |            2610.74 |                2.48098 |
| 49743 | chevrolet     | matiz        |                2005 |          1000 | το αυτοκινητο κινητε κανονικα ειναι για ανταλλακτι |         840 |            2895.61 |                2.44716 |
| 44707 | skoda         | fabia        |                2006 |          1400 | nan                                                |        1790 |            6150.91 |                2.43626 |
| 47957 | alfa romeo    | alfa 156     |                2002 |          1600 | θελει καποια μαζεματα το αμαξι δουλευει κανονικα ε |         700 |            2396.7  |                2.42386 |


### Reliability of the estimation:

We want to know how reliable is the model's prediction. And we want to be able to give a low price and a high price to the user. In order to do this we trained 3 catboost models which estimates the median price, one which estimates the first quartile and one which estimates the third quartile. This enables us to get a range as well as an estimation. To train the q1 and q3 model we used the same hyperparameters found for q2 for the sake of simplicity.



Now we can consider that the larger the range is with respect to the estimated price, the more uncertain the estimation is. Based on this idea we construct a reliability metric. We could calculate the difference between q3 and q1 normalised by the estimated price, however this measure is unbounded and therefore it is difficult to interpret for the user. Ideally we would like to have an indicator between 0 and 1 wher 1 means very reliable and 0 unreliable. 

For this we can use a sigmoid function on the normalised interquartile range of the estimation. of the form 

![equation(1)](https://github.com/user-attachments/assets/b688d563-87ef-45b0-966c-9edd08db4e9d)

where:
- \(q_1\) represents the first quartile estimation,
- \(q_2\) represents the median estimation,
- \(q_3\) represents the third quartile estimation
- \(c\) is the value on which the sigmoid is centered,
- \(\lambda\) (lambda) is a shape parameter controlling the sensitivity of the function.

We can interpret the ratio (q3 - q1)/q2 as the size of the IQR as a percentage of the median estimation. The constant c represents the percentage size of the IQR which we consider to be a score of 0.5. Lambda determines how steep the sigmoid function is around c and how fast it gets close to 0 or 1. 

If we look at the describe of the percentage size of the estimated IQR with respect to the median we see that the median uncertainty equals 15%, that means when the model estimates a price with +/- 7.5% certainty. We see that the 90th percentile equals 34%, therefore we will want the score to reach a score close to 0 at around 90%. 

![image](https://github.com/user-attachments/assets/80c077f5-9348-4bc1-a0b6-f2c5b19032e5)


We decide that we will consider a score of 0.5 when the level of uncertainty is of +/- 10% that is for c = 0.2 (20% of the median estimation).  We will consider the score reliable when the reliability score is above 75%, we will consider it unreliable when between 25% and 75%, and very unreliable when below 25%. 

![reliability_score](https://github.com/user-attachments/assets/d48c99f7-9cdc-41d6-a9a4-f8aeb3c112b5)


![boxplot_reliability_per_year](https://github.com/user-attachments/assets/b8c0589c-41c1-4e7a-b161-afd863f3f373)

![reliability_by_brand](https://github.com/user-attachments/assets/b47e98a5-567a-4a81-a27b-67e013386880)


50% of the data has a reliability below 73%. 

![image](https://github.com/user-attachments/assets/02bacb5b-4818-4942-836b-0d845c6b2129)

![image](https://github.com/user-attachments/assets/53f12a55-b7e2-4c2b-aa6b-bebfe79c02f6)

![image](https://github.com/user-attachments/assets/45a9cf3e-6ef9-422c-8323-22e206b0f5ef)


## Interpretability:

Feature importances:

Below we plot the feature importances of the chose catboost model: 


![feature_importances_catboost](https://github.com/user-attachments/assets/721ae31b-b28d-4b02-aa99-c8b6edecaa9f)

We see that the most important features are the engine size, power, brand, model, gearbox type, mileage, registration year.  

In order for the user to get an idea of the way features such as mileage, engine power or extras would affect the price we display the Individual COnditional Expectation (ICE) plot which shows the predicted price for a range of values of the analysed feature, keeping all others fixed.

As an example, the user here estimates a suzuki swift with 150000km from 2015, with 1200cc engine size, 70bhp. 
![image](https://github.com/user-attachments/assets/e41bcc08-1225-4e27-afa3-85809f1746ee)

If the user wants to see how mileage affects the price, we will plot the predicted price and IQR for this car making the value of mileage vary from 0 to a large value:

![image](https://github.com/user-attachments/assets/efc987c9-fbed-492e-899d-dd08e2b9b244)


It is interesting to see that the model becomes uncertain for unusual large engine sizes as the suzuki swift is a small car: 

![image](https://github.com/user-attachments/assets/83532c93-188a-40fd-b8b9-a5a0b182eedb)



## Deployment

The model was deployed on google cloud run as a streamlit app containerised on google container registry. The model and experiments for training were run on a vm instance on google cloud platform in order to leverage catboost gpu compatibility. The streamlit app allows the user to input all the features taken into account by the model and to estimate the price and price range and reliability score of the car. It then allows the user to visualise effects of all the variables taken into account by the model. 

The app gets redeployed everytime code is pushed in the main branch via git actions. 

## Conclusion, challenges, weaknesses and potential improvements


- Some models were listed at a very granular level on the website, for example mercedes A180, A 160, A200 instead of just mercedes ACLASS. The same thing is observed on various other brands with specs determining the model name such as bmws. This leads to certain models being sparse in the data set and regrouping them under the same class could have helped the model leverage the model feature more effectively
- The condition of the car is very important and we haven't really taken that into account apart from the crashed boolean feature. In fact we saw multiple cars being sold for spare parts which had very low prices and identifying such cases could help clean the data more easily and also improve model performance. We could fine tune a transformer to classify car descriptions into Very good condition, good, normal, bad, very bad, unknown for example. We coudl also do it using the pictures by fine tuning a vision transformer. However this would require a labeled dataset, something which we did not have the time to construct for now.
- Geolocation features weren't studied until now and it might be important to. In fact the car market could be different in different regions in greece? We did not focus on this as we assumed that prices shouldn't differ much accross the territory but this still needs to be verified
- The Dataset was constructed from the scraping done in February. This gives us a frozen image of the car market in february 2025 and this means that model will not be able to effectively estimate car prices as precisely in a few months. In a real production environment, for the model to stay accurate and avoid data drift over time we should rescrape the website regularly and retrain the model on the new data, verifying that performance doesn't degrade over time.
- We did not take the ad posting date or modification date into account as we only scraped the website once. However keeping track of ad history and price changes in a database could help take an additional temporal dimension into account which could be important. This is something we didn't take into account.
- More advanced outlier removal strategies such as isolation forest, clustering algorithms could have been tried.
- More advanced feature engineering techniques on extras for example or extracting additional information from text data could have been implemented.
- More models could have been tried, for example generalised additive models seem like an ideal method as well for this type of data.
- Extracting features from images to enrich the data using computer vision algorithms could help impute missing data

The nature of the data and the time it took to scrape, clean and preprocess it until now did not allow me to explore all these factors deep enough but it remains something that I wish to do in the future.

Thank you for your attention and looking forward to any feedback or insights you may have about the project. 

