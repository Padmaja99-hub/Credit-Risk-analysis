# Credit-Risk-analysis
This project is based on the publicly accessible Credit Risk Dataset which is available on Kaggle.

Problem Statement:
In the competitive lending market, loan defaults can pose a significant risk to financial stability and profitability. The problem I aim to solve in this analysis is identifying the characteristics and behaviors of borrowers most likely to default on loans. This analysis will inform risk-mitigation strategies, such as setting appropriate interest rates, defining loan limits, and establishing tailored approval criteria. Loan defaults affect both lenders and borrowers. By analyzing borrower profiles and default patterns, I aim to develop lending practices that balance fairness and risk aversion, contributing to a more sustainable financial system.

Key Questions:
What factors are most strongly associated with loan defaults?
How can we predict the likelihood of default for individual borrowers?
What thresholds or policies can minimize risk while ensuring loan accessibility?
Data Overview:
This dataset contains columns simulating credit bureau data. The dataset consists of 32,581 rows and 12 columns. Key columns include: Person_income, Loan_int_rate, loan_percent_income, cb_person_cred_hist_length etc.

Tools Used:

• Python (pandas, numpy, matplotlib, seaborn, sklearn, xgboost)

Data Cleaning and Preprocessing:
Handling Null Values:

• 'person_emp_length': 2.7% missing values were dropped.

image
• 'loan_int_rate': 9.5% missing values imputed based on 'loan_grade' to preserve data quality.

image
Feature Engineering:

• New Feature: 'has_income_no_emp': Flags borrowers with income but no employment length. This distinguishes new workers or self-employed individuals from traditional employees.

image
• Categorization: Grouped 'person_age' into bins and segmented 'person_income' into ranges.

image image
Removing Outliers:

• Unrealistic entries like 'person_age' (144 years) and 'person_emp_length' (123 years) were removed to improve data reliability.

• Person_emp_length: Max entry showing 123 years. It's unrealistic for an individual to have worked for such an extended period, and this could be due to data entry errors or inconsistencies. I will delete the entries.

image
I exported clean data to ensure accuracy and consistency for analysis and reporting. My data is now ready for EDA.

image
Exploratory Data Analysis (EDA):
Univariate Analysis : Categorical & Numerical :

image image image image image image
Key Insights:

• Most borrowers are young (16-45 years) with limited employment length (0-10 years), suggesting a borrower pool of early-career individuals.

• Renters and mortgage payers dominate, indicating that borrowers are primarily those with regular housing expenses.

• Educational and medical loans are the most common, reflecting financial needs for essential purposes.

• Non-defaulters vastly outnumber defaulters, showing that most borrowers manage to repay loans successfully.

• Borrowers with incomes between $50,000-$100,000 form the largest group, representing economically active individuals. Interest rates are primarily clustered around 7.5%, with an average of 11.03%, and goes as high as 22.5%.

• A low loan-to-income ratio (0-0.2%) is prevalent, indicating that affordability is a key consideration for most loans.

• Many borrowers have short credit histories (0-5 years), reflecting their younger age and limited financial experience.

Bivariate Analysis :

image image image image image

image image image image image

Key Insights:

The Bivariate analysis reveals key patterns in borrower behavior related to loan defaults:

• Mortgage payers are more likely to repay their loans, while rent payers have higher default rates.

• Borrowers with educational loans are more likely to repay than those with medical loans, indicating that education-related loans carry lower risk.

• Loans with grades D–G show higher default rates, while A-grade loans tend to have more reliable repayment histories.

• Borrowers earning less than $50,000 or with less than 5 years of employment history are more likely to default, suggesting that financial stability plays a crucial role in loan repayment.

• Loans with interest rates above 15.4% and high loan-to-income ratios are associated with higher default rates, indicating that higher-risk loans are more likely to default.

Multivariate Analysis:

=> Correlation matrix:

image

Key Insights:

• Loan_int_rate correlates with loan_grade and previous default history.

• Loan_percent_income is strongly predictive of loan status, suggesting that the proportion of income spent on loan repayments is a significant factor in determining loan default risk.

• Age correlates positively with credit history length, indicating borrowers build credit history over time.

• There is a weak but notable relationship between renters and borrowers with zero employment length, implying that renters with no employment history may present higher risk.

Variables most strongly correlated with loan status:

a. Loan_percent_income

b. Loan interest rate

c. Loan grade D

d. Person home ownership – Rent

e. Loan grade E

f. Previous default history

Investigating strongly correlated variables:

• Loan percentage income:

image
Threshold set at .31

=> Loan percent income ratios exceeding 0.31 show a default rate of 70.32%, indicating significant financial stress among these borrowers.

=> Default rate drops to 25.1% for loan percent income below threshold.

A higher loan percent income suggests that the borrower is spending a larger portion of their income on repaying the loan, which could indicate higher financial stress. The high default rate (70.32%) among borrowers with loan percent income above 0.31 suggests that loans that represent a larger proportion of income are riskier. This finding can guide lending policies, where borrowers with higher loan-to-income ratios may need additional scrutiny or higher interest rates to account for the increased risk

• Loan Interest Rate:

image
A threshold of 15.4% was established.

=> Default rate for interest rates exceeding this threshold is 61.2%.

=> For interest rates below the threshold, the default rate drops to 17.8%.

The correlation matrix reveals that the loan interest rate is strongly associated with the loan percentage income, prior default history, and loan grade.

• Income and Age Relationship:

image
a. Borrowers with incomes below $5,000 exhibit consistently high default rates across all age groups, highlighting income as a more significant risk factor than age. b. The high default rates in the $0–$5,000 income range suggest that low income is a strong indicator of loan risk, regardless of the borrower's age. c. Segmenting by age adds minimal predictive value since default rates remain consistently high among low-income borrowers across all age groups.

• Renters with Zero Employment Length:

image
Out of 2,666 renters with zero employment length, 929 are defaulting, highlighting a default rate of approximately 34.9% for this high-risk .

Predictive model:
While some features in my dataset, person_income and person_age, exhibit skewness. But as I am primarily using tree based models,transformation is not necessary for my analysis. Tree-based models such as XGBoost and Random Forest, which are robust to skewed data, effectively handle raw features without requiring normalization. Feature Engineering and Selection:

Before modeling, I focused on enhancing the dataset with engineered features that capture high-risk behaviors and simplifying it by removing redundant information. The goal was to ensure the model learns meaningful patterns while avoiding overfitting. In this step, I created a composite risk score to better assess borrower risk by combining multiple financial factors, such as loan interest rate, loan grade, income percentage, home ownership, and prior default history. These factors were mapped to numerical values and scaled using MinMax scaling to ensure equal contribution to the score. The composite risk score was then categorized into four risk levels: Low, Medium, High, and Very High. This approach provides a more nuanced view of borrower risk by integrating multiple features, helping to make more informed decisions about loan approval and potential defaults.

image
Dropping Redundant Features:

• Removed intermediate flags and features like age_group and income_range that were either captured within engineered features or did not add unique predictive value.

Modeling and Predictions:

Algorithms Used:

• Random Forest: Best for precision

• KNNEighbour Classifier

• Logistic Regression

• DecisionTree Classifier

• XGBoost Classifier: Best for recall

Model Performance:

image image
• XGBoost outperformed in recall, identifying 35 more defaults, making it better at capturing true positives.

• Random Forest excelled in precision, reducing false positives by 18%, ensuring more accurate predictions.

Feature Importance:

Random Forest provides insights into feature importance by measuring how much each feature contributes to reducing impurity across all trees. Features with higher importance scores have a stronger influence on predictions, helping identify key drivers for the model's output.

image
Recommendations:
Target Lower-Grade Loan Risk: Implement stricter approval criteria or support mechanisms for D–F grade loans.

Support Medical Loan Borrowers: Offer financial counseling and flexible repayment options.

Focus on Income and Employment History: Evaluate job stability or additional income sources for borrowers with low incomes or short employment histories.

Cap High-Interest Loans: Limit loans with interest rates above the 15.4% threshold or offer tailored products to mitigate default risk.

Challenges Faced:
• Managing imbalanced data due to a higher proportion of non-defaulters.

• Imputing missing values without introducing bias.

• Deciding thresholds for features like loan-to-income ratio and interest rates.

Lessons Learned:

• Feature engineering is critical in enhancing model accuracy and interpretability.

• High-dimensional datasets require careful correlation analysis to avoid multicollinearity.

• Correlation ≠ Causation: Always validate assumptions through multivariate analysis.

• Removing outliers and imputing values significantly improves model reliability.
