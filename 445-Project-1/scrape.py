
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.webdriver import WebDriver as Chrome
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

# Set up Chrome options
chrome_options = Options()
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# Initialize ChromeDriver service
service = Service('chromedriver.exe')

# Run Selenium with ChromeDriver
driver = Chrome(service=service, options=chrome_options)

# Open job listings page
driver.get('https://www.simplyhired.com/search?q=software+engineer&l=New+York%2C+NY&job=GuNt4GJc2YBrfsww0U6mjHEOwza-HhqkZHos7BIg1YMXJZRybiyawA')

# Function to handle cookie pop-ups
def handle_cookies():
    try:
        WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Accept')]"))
        ).click()
    except Exception as e:
        print(f"Cookies statement not found or could not be clicked: {e}")

    try:
        WebDriverWait(driver, 10).until(
            EC.invisibility_of_element((By.XPATH, "//div[@id='onetrust-policy-text']"))
        )
    except Exception as e:
        print(f"Cookie banner did not disappear: {e}")

# Wait for job listings to load
WebDriverWait(driver, 10).until(
    EC.presence_of_all_elements_located((By.CLASS_NAME, "css-obg9ou"))
)

# Scroll the page to load more listings
def scroll_page():
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, "css-obg9ou"))
        )
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

# List to store job data
job_listings = []

# Function to scrape job listings
def scrape_listings():
    listings = driver.find_elements(By.CLASS_NAME, "css-obg9ou")
    print(f"Number of listings found: {len(listings)}")

    for listing in listings:
        try:
            job_title_element = listing.find_element(By.CSS_SELECTOR, 'a.chakra-button.css-1djbb1k')
            company_element = listing.find_element(By.CSS_SELECTOR, "span.css-lvyu5j span[data-testid='companyName']")
            location_element = listing.find_element(By.CLASS_NAME, "css-1t92pv")

            title = job_title_element.text
            company = company_element.text
            location = location_element.text
            qualifications = []

            try:
                # Click job listing to extract additional details
                job_title_element.click()
                WebDriverWait(driver, 20).until(
                    EC.presence_of_all_elements_located((By.CLASS_NAME, "css-p3sbg2"))
                )
                qualifications_elements = driver.find_elements(By.CLASS_NAME, "css-p3sbg2")
                qualifications = [elem.text for elem in qualifications_elements]
            except Exception as e:
                print(f"Error clicking: {e}")

            job_listings.append({
                'title': title,
                'company': company,
                'location': location,
                'qualifications': qualifications
            })
        except Exception as e:
            print(f"Error processing listing: {e}")

# Paginate through listings
while True:
    scrape_listings()
    try:
        next_button = driver.find_element(By.CSS_SELECTOR, "#__next > div > main > div > div.css-17iqsqz > div > div > div.css-2jn6zr > div > div.css-15g2oxy > div.css-ukpd8g > nav > a.chakra-link.css-1puj5o8")
        print(f"Next button found: {next_button.text}")
        if next_button:
            click_success = False
            for attempt in range(3):
                try:
                    driver.execute_script("arguments[0].click();", next_button)
                    WebDriverWait(driver, 20).until(
                        EC.presence_of_element_located((By.CLASS_NAME, "css-obg9ou"))
                    )
                    click_success = True
                    break
                except Exception as e:
                    print(f"Attempt {attempt + 1} to click 'Next' button failed: {e}")
            if not click_success:
                break
        else:
            break
    except Exception as e:
        print(f"No more pages or error: {e}")
        break

# Close the driver after scraping
driver.quit()

# Convert scraped data to DataFrame
job_listings_df = pd.DataFrame(job_listings)
print(job_listings_df.head())
total_rows = len(job_listings_df)
print(f"Total rows before preprocessing: {total_rows}")

# Save to CSV file
job_listings_df.to_csv('job_listings.csv', index=False)
print("Job Listings have been saved to job_listings.csv")

# Step 1: Read the CSV file
job_listings_df = pd.read_csv('job_listings.csv')

# Step 2: Feature Engineering
def extract_skills(qualifications):
    if qualifications == 'Unknown' or pd.isna(qualifications):
        return ["no skills"]
    try:
        qual_list = eval(qualifications)
        skills = []
        for qual in qual_list:
            if not any(keyword in qual for keyword in ['experience', 'years', 'mid-level', 'senior']):
                skills.append(qual)
        if not skills:
          return ["no skills"]
        return skills
    except (SyntaxError, TypeError, ValueError):
        return ["no skills"]

def extract_experience(qualifications):
    if qualifications == 'Unknown' or pd.isna(qualifications):
        return "Not Specified"
    try:
        qual_list = eval(qualifications)
        for qual in qual_list:
            if any(keyword in qual for keyword in ['experience', 'years', 'mid-level', 'senior']):
                return qual
        return "Not Specified"
    except (SyntaxError, TypeError, ValueError):
        return "Not Specified"

job_listings_df['required_skills'] = job_listings_df['qualifications'].apply(extract_skills)
job_listings_df['experience_level'] = job_listings_df['qualifications'].apply(extract_experience)

# Step 3: Encode Categorical Features and Skills
def encode_features(df):
    vectorizer = CountVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x)
    skills_matrix = vectorizer.fit_transform(df['required_skills'])
    skills_df = pd.DataFrame(skills_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    df = pd.concat([df.reset_index(drop=True), skills_df.reset_index(drop=True)], axis=1)

    le = LabelEncoder()
    df['title_encoded'] = le.fit_transform(df['title'])

    df = pd.get_dummies(df, columns=['location', 'experience_level'], drop_first=True, dummy_na=True)
    df.drop(columns=['required_skills', 'title'], inplace=True)
    return df, skills_matrix, vectorizer

encoded_df, skills_matrix, vectorizer = encode_features(job_listings_df)

X = encoded_df.drop(columns=['company', 'qualifications'])
y = encoded_df['title_encoded']

# Step 4: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Scale the Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train RandomForestClassifier to Determine Feature Importance
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train_scaled, y_train)

# Step 7: Feature Importance
feature_importance = rf_clf.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Step 8: Visualization
# Feature Importance Visualization (Improved)
plt.figure(figsize=(14, 8))
# Exclude 'title_encoded' from visualization
filtered_importance = importance_df[importance_df['Feature'] != 'title_encoded'].head(15)
sns.barplot(data=filtered_importance, x='Importance', y='Feature', palette='viridis')
plt.title('Top 15 Important Features in Job Roles (Excluding Title)', fontsize=16)
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print(filtered_importance)

# Visualize Skill Distribution
skills_data = pd.DataFrame(skills_matrix.toarray(), columns=vectorizer.get_feature_names_out())
skill_counts = skills_data.sum().sort_values(ascending=False).head(20)
#Exclude "no skills" from skill visualization.
skill_counts = skill_counts[skill_counts.index != "no skills"]
plt.figure(figsize=(14, 7))
sns.barplot(x=skill_counts.values, y=skill_counts.index, palette='coolwarm')
plt.title('Top Skills Distribution (Excluding "no skills")', fontsize=16)
plt.xlabel('Count', fontsize=12)
plt.ylabel('Skill', fontsize=12)
plt.tight_layout()
plt.show()

# Visualize Experience Level Distribution
plt.figure(figsize=(8, 6))
experience_counts = job_listings_df['experience_level'].value_counts()
#Exclude "Not Specified" from experience visualization.
experience_counts = experience_counts[experience_counts.index != "Not Specified"]
sns.barplot(x=experience_counts.index, y=experience_counts.values, palette='Set2')
plt.title('Experience Level Distribution (Excluding "Not Specified")', fontsize=16)
plt.xlabel('Experience Level', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Visualize Location Distribution
plt.figure(figsize=(10, 6))
location_counts = job_listings_df['location'].value_counts()
sns.barplot(x=location_counts.index, y=location_counts.values, palette='pastel1')
plt.title('Location Distribution', fontsize=16)
plt.xlabel('Location', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
