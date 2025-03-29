# Credit-Risk-Data-Analysis
This project is based on the publicly accessible [Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) which is available on [Kaggle](https://www.kaggle.com/).



# Problem Statement:

In the competitive lending market, loan defaults can pose a significant risk to financial stability and profitability. The problem I aim to solve in this analysis is identifying the characteristics and behaviors of borrowers most likely to default on loans. This analysis will inform risk-mitigation strategies, such as setting appropriate interest rates, defining loan limits, and establishing tailored approval criteria. Loan defaults affect both lenders and borrowers. By analyzing borrower profiles and default patterns, I aim to develop lending practices that balance fairness and risk aversion, contributing to a more sustainable financial system.

# Key Questions:

1.	What factors are most strongly associated with loan defaults?
2.	How can we predict the likelihood of default for individual borrowers?
3.	What thresholds or policies can minimize risk while ensuring loan accessibility?
   
# Data Overview:

This dataset contains columns simulating credit bureau data. The dataset consists of 32,581 rows and 12 columns. Key columns include: Person_income, Loan_int_rate, loan_percent_income, cb_person_cred_hist_length etc.

**Tools Used:**

•	 Python (pandas, numpy, matplotlib, seaborn, sklearn, xgboost)


# Data Cleaning and Preprocessing:
   
**Handling Null Values:**

•	'person_emp_length': 2.7% missing values were dropped.

<img width="436" alt="image" src="https://github.com/user-attachments/assets/54c639ff-1d5a-4aca-b75a-59f29e97c458">


•	'loan_int_rate': 9.5% missing values imputed based on 'loan_grade' to preserve data quality.

<img width="598" alt="image" src="https://github.com/user-attachments/assets/7530985b-458d-4c7c-a218-e668d8c33486">



**Feature Engineering:**

•	New Feature: 'has_income_no_emp': Flags borrowers with income but no employment length. This distinguishes new workers or self-employed individuals from traditional employees.

<img width="516" alt="image" src="https://github.com/user-attachments/assets/09c86192-309a-4733-9e8f-0d9fec88c6f5">


•	Categorization: Grouped 'person_age' into bins and segmented 'person_income' into ranges.

<img width="439" alt="image" src="https://github.com/user-attachments/assets/84853079-017b-4836-9ede-c103f89d9bc6">
<img width="723" alt="image" src="https://github.com/user-attachments/assets/33726387-1c3e-48ff-b2aa-82b17fc752d2">


 **Removing Outliers:**

•  Unrealistic entries like 'person_age' (144 years) and 'person_emp_length' (123 years) were removed to improve data reliability.

•  Person_emp_length: Max entry showing 123 years. It's unrealistic for an individual to have worked for such an extended period, and this could be due to data entry errors or inconsistencies. I will delete the entries.

<img width="419" alt="image" src="https://github.com/user-attachments/assets/f7922d9f-3f5f-4b3b-97c7-176bf05f19eb">

I exported clean data to ensure accuracy and consistency for analysis and reporting. My data is now ready for EDA.

<img width="307" alt="image" src="https://github.com/user-attachments/assets/8eaa4a27-0ecc-48da-b3a6-366d90cf016b">


# Exploratory Data Analysis (EDA):

**Univariate Analysis :  Categorical & Numerical :**

<img width="424" alt="image" src="https://github.com/user-attachments/assets/e159da73-47a6-49b2-a48d-1acbaa988c84">

<img width="737" alt="image" src="https://github.com/user-attachments/assets/b5917c4f-e748-44ea-bf59-90dd38f84a17">
<img width="738" alt="image" src="https://github.com/user-attachments/assets/d2b3103e-730a-472b-a088-312767b0027b">
<img width="731" alt="image" src="https://github.com/user-attachments/assets/4493bc5b-f587-4715-a75f-445e19ae33b0">
<img width="733" alt="image" src="https://github.com/user-attachments/assets/dae5386c-4f8c-455e-b2a5-46f5ed159607">
<img width="734" alt="image" src="https://github.com/user-attachments/assets/aa394f3f-5dd7-4651-a981-3bf3efd5885d">


**Key Insights:**

•	Most borrowers are young (16-45 years) with limited employment length (0-10 years), suggesting a borrower pool of early-career individuals.

•	Renters and mortgage payers dominate, indicating that borrowers are primarily those with regular housing expenses.

•	Educational and medical loans are the most common, reflecting financial needs for essential purposes.

•	Non-defaulters vastly outnumber defaulters, showing that most borrowers manage to repay loans successfully.

•	Borrowers with incomes between $50,000-$100,000 form the largest group, representing economically active individuals. Interest rates are primarily clustered around 7.5%, with an average of 11.03%, and goes as high as 22.5%.

•	A low loan-to-income ratio (0-0.2%) is prevalent, indicating that affordability is a key consideration for most loans.

•	Many borrowers have short credit histories (0-5 years), reflecting their younger age and limited financial experience.


**Bivariate Analysis :**


<img width="376" alt="image" src="https://github.com/user-attachments/assets/ad3140ab-002e-44b5-8a60-0fbafa75b84b"> <img width="383" alt="image" src="https://github.com/user-attachments/assets/ae4cb3ee-292f-4ea9-8b1f-bf34e5ca468a"> <img width="374" alt="image" src="https://github.com/user-attachments/assets/59cef8e5-46ca-4d6b-9134-7847fe96f50b">  <img width="374" alt="image" src="https://github.com/user-attachments/assets/a7ffac94-f2d3-47d3-88f4-0a239d20c79f">  <img width="386" alt="image" src="https://github.com/user-attachments/assets/c2449a8f-083e-451d-a17c-25c16e3cb985">


<img width="370" alt="image" src="https://github.com/user-attachments/assets/bf6e9dd0-77f3-4dc9-8b52-26ba2e1cd5f3"> <img width="399" alt="image" src="https://github.com/user-attachments/assets/428d232d-28eb-4f38-8136-a610c0edf8a0"> <img width="373" alt="image" src="https://github.com/user-attachments/assets/1930ffc0-5275-4292-99e8-7ecd24b8ce0a"> <img width="362" alt="image" src="https://github.com/user-attachments/assets/55f57a1e-3e26-42ef-9cc0-7b2adf7eb73f">  <img width="374" alt="image" src="https://github.com/user-attachments/assets/26746bbf-de05-4851-93ad-f4f4cb8d8e00">


**Key Insights:**

The Bivariate analysis reveals key patterns in borrower behavior related to loan defaults:

•	Mortgage payers are more likely to repay their loans, while rent payers have higher default rates.

•	Borrowers with educational loans are more likely to repay than those with medical loans, indicating that education-related loans carry lower risk.

•	Loans with grades D–G show higher default rates, while A-grade loans tend to have more reliable repayment histories.

•	Borrowers earning less than $50,000 or with less than 5 years of employment history are more likely to default, suggesting that financial stability plays a crucial role in loan repayment.

•	Loans with interest rates above 15.4% and high loan-to-income ratios are associated with higher default rates, indicating that higher-risk loans are more likely to default.

**Multivariate Analysis:**

=> Correlation matrix:

![image](https://github.com/user-attachments/assets/f50d4d44-231f-4eaf-80ef-5a8d89efdeda)


**Key Insights:**
 
•	Loan_int_rate correlates with loan_grade and previous default history.

•	Loan_percent_income is strongly predictive of loan status, suggesting that the proportion of income spent on loan repayments is a significant factor in determining loan default risk.

•	Age correlates positively with credit history length, indicating borrowers build credit history over time.

•	There is a weak but notable relationship between renters and borrowers with zero employment length, implying that renters with no employment history may present higher risk.


**Variables most strongly correlated with loan status:**

a.	Loan_percent_income 

b.	Loan interest rate 

c.	Loan grade D 

d.	Person home ownership – Rent 

e.	Loan grade E 

f.	Previous default history 


**Investigating strongly correlated variables:**

**• Loan percentage income:**

<img width="341" alt="image" src="https://github.com/user-attachments/assets/c34fde4e-403b-4d49-acbf-a4b6679c7fad">


Threshold set at .31

  =>	Loan percent income ratios exceeding 0.31 show a default rate of 70.32%, indicating significant financial stress among these borrowers.
  
  =>	Default rate drops to 25.1% for loan percent income below threshold.

A higher loan percent income suggests that the borrower is spending a larger portion of their income on repaying the loan, which could indicate higher financial stress. The high default rate (70.32%) among borrowers with loan percent income above 0.31 suggests that loans that represent a larger proportion of income are riskier. This finding can guide lending policies, where borrowers with higher loan-to-income ratios may need additional scrutiny or higher interest rates to account for the increased risk


   
**•	Loan Interest Rate:**

<img width="304" alt="image" src="https://github.com/user-attachments/assets/ef8c515b-4e7b-4abe-a51a-8905ae1ccf99">

A threshold of 15.4% was established.

  =>	Default rate for interest rates exceeding this threshold is 61.2%.
  
  =>	For interest rates below the threshold, the default rate drops to 17.8%.
  
The correlation matrix reveals that the loan interest rate is strongly associated with the loan percentage income, prior default history, and loan grade.

   
•	Income and Age Relationship: 

<img width="227" alt="image" src="https://github.com/user-attachments/assets/14fbe0f5-3915-41ef-8938-e16fb8cc3ebd">


a.	Borrowers with incomes below $5,000 exhibit consistently high default rates across all age groups, highlighting income as a more significant risk factor than age.
b.	The high default rates in the $0–$5,000 income range suggest that low income is a strong indicator of loan risk, regardless of the borrower's age.
c.	Segmenting by age adds minimal predictive value since default rates remain consistently high among low-income borrowers across all age groups.

**• Renters with Zero Employment Length:**

<img width="289" alt="image" src="https://github.com/user-attachments/assets/3cbde2f1-12b9-4b79-acd1-c43dec63ec3a">


Out of 2,666 renters with zero employment length, 929 are defaulting, highlighting a default rate of approximately 34.9% for this high-risk .

# Predictive model:

While some features in my dataset, person_income and person_age, exhibit skewness. But as I am primarily using tree based models,transformation is not necessary for my analysis. Tree-based models such as XGBoost and Random Forest, which are robust to skewed data, effectively handle raw features without requiring normalization.
Feature Engineering and Selection:

Before modeling, I focused on enhancing the dataset with engineered features that capture high-risk behaviors and simplifying it by removing redundant information. The goal was to ensure the model learns meaningful patterns while avoiding overfitting.
In this step, I created a composite risk score to better assess borrower risk by combining multiple financial factors, such as loan interest rate, loan grade, income percentage, home ownership, and prior default history. These factors were mapped to numerical values and scaled using MinMax scaling to ensure equal contribution to the score. The composite risk score was then categorized into four risk levels: Low, Medium, High, and Very High. This approach provides a more nuanced view of borrower risk by integrating multiple features, helping to make more informed decisions about loan approval and potential defaults.

<img width="730" alt="image" src="https://github.com/user-attachments/assets/535b2fbd-f646-4622-9e2d-fe9d5ea7daff">


**Dropping Redundant Features:**

•	Removed intermediate flags and features like age_group and income_range that were either captured within engineered features or did not add unique predictive value.

**Modeling and Predictions:**

Algorithms Used:

•	Random Forest: Best for precision 

•	KNNEighbour Classifier

•	Logistic Regression

•	DecisionTree Classifier

•	XGBoost Classifier: Best for recall 
	
**Model Performance:**

<img width="274" alt="image" src="https://github.com/user-attachments/assets/2061d447-6bea-49c2-97b9-b7ae817a60ed">

<img width="257" alt="image" src="https://github.com/user-attachments/assets/ac1f02f7-68ec-474e-b4f0-d8edeff4f379">


•	XGBoost outperformed in recall, identifying 35 more defaults, making it better at capturing true positives.

•	Random Forest excelled in precision, reducing false positives by 18%, ensuring more accurate predictions.


**Feature Importance:**

Random Forest provides insights into feature importance by measuring how much each feature contributes to reducing impurity across all trees. Features with higher importance scores have a stronger influence on predictions, helping identify key drivers for the model's output.

<img width="230" alt="image" src="https://github.com/user-attachments/assets/8f2d6b33-33af-4baa-80e5-52c5383fce1e">

# Recommendations:

1.	Target Lower-Grade Loan Risk:	Implement stricter approval criteria or support mechanisms for D–F grade loans.
   
2.	Support Medical Loan Borrowers:	Offer financial counseling and flexible repayment options.
	
3.	Focus on Income and Employment History: Evaluate job stability or additional income sources for borrowers with low incomes or short employment histories.
	
4.	Cap High-Interest Loans: Limit loans with interest rates above the 15.4% threshold or offer tailored products to mitigate default risk.

   
# Challenges Faced:

•	Managing imbalanced data due to a higher proportion of non-defaulters.

•	Imputing missing values without introducing bias.

•	Deciding thresholds for features like loan-to-income ratio and interest rates.


**Lessons Learned:**

•	Feature engineering is critical in enhancing model accuracy and interpretability.

•	High-dimensional datasets require careful correlation analysis to avoid multicollinearity.

•	Correlation ≠ Causation: Always validate assumptions through multivariate analysis.

•	Removing outliers and imputing values significantly improves model reliability.
