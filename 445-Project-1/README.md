# Description of the Project

This project aims to analyze salary trends for Software Engineers in different locations in order to see how developers with different skills are paid-- and which of those skills are best compensated/desired.

# How to Use

- Clone repo and install the dependencies that you need as indicated by your environment.

## Training

The machine learning models are trained using data scraped from an online job board, "simplyhired.com".

One-hot encoding was used to extract features that are needed for the model.

A RandomForestClassifier and other models are trained on labeled datasets to helpo predict the salary ranges.

## Inferences

Given a new job description (skills and location), the trained models predict the expected salary range based on the data that is given.

The models could also be used to identify the most important skills required for a job.

# Data Collection

## Used Tools

- Selenium: I used it for web scraping job postings from SimplyHired.

- Pandas: Only used for data processing and storage.

- Scikit-learn: This makes it easier to develop the actual models.

## Data Sources

Job listings scraped from SimplyHired.

Extracted attributes include job title, company, location, skills, and salary range.

## Collected Attributes

- Job title

- Company name

- Location

- Required skills

- Experience level

- Salary (if available)

## Number of Data Samples

Approximately 850 job listings were collected, which took a surprising amount of time.

Data Preprocessing

# Data Cleaning

Removed duplicate job listings.

## Data Description and Metadata Specification

Data repository (CSV) contains structured job postings.

## Sample data
![image](https://github.com/user-attachments/assets/4e4cb647-03dc-4541-b23e-247ad4d90324)

# Feature Engineering

- Extracted skills from job descriptions using NLP techniques.

- Applied CountVectorizer to encode skills as numerical features.

- One-hot encoded categorical variables such as location and experience level.

# Model Development and Evaluation

## Machine Learning Model

RandomForestRegressor

Input to Model

Encoded job descriptions, skills, and experience levels.

The size of the training data is around 850.

## Attributes to the Machine Learning Model

Job title, experience level, location, and extracted skills.

# Project Findings

Other than the obvious locations based on our search, one can see that knowledge of Restful API's seem to be very important in Job Roles for the local market.
![image](https://github.com/user-attachments/assets/eaf588d1-2147-429a-8160-45d07b5e9321)

AWS seems to be leading the way in terms of what the most amount of Employers want to see in thier potential recruit's skill section.
![image](https://github.com/user-attachments/assets/d2c90ca6-c401-431b-b953-77a16c85fbf4)

A trend can be seen that implies that employers prefer mid level employees, then senior level, then entry level in order of priority.
![image](https://github.com/user-attachments/assets/a18cfbe3-87bb-4c85-9848-849eafd3329f)

# Challenges Encountered

By far the hardest part was figuring out how to scrape and organize the data itself. Not only is it a hard prospect in the first place, but with things websites put in place to stop scraping-- it makes it extremely difficult to get any amount of usable data through methods like that.

# Recommendations

- Better data cleaning.

- Improvement on models.

- Overall robustness.
