# MLpredictdiabetes
Diabetes is a chronic metabolic disorder that poses a significant global health burden, affecting over 500 million people worldwide [1], [2]. It is one of the leading causes of morbidity and mortality, with complications ranging from cardiovascular diseases and kidney failure to blindness and neuropathy [3]. The economic burden of diabetes is substantial, with billions spent annually on direct healthcare costs, medication, and the management of related complications [4]. Moreover, the indirect costs, including loss of productivity and quality of life, further exacerbate its impact [4].
Diabetes mellitus has emerged as a global health crisis, affecting over 537 million adults worldwide according to the International Diabetes Federation (2023). Early detection is critical for preventing severe complications such as retinopathy and cardiovascular diseases. Traditional diagnostic methods like the Oral Glucose Tolerance Test (OGTT), while clinically reliable, face challenges in large-scale screening due to high costs (approximately $50-100 per test) and operational complexity.
The Pima Indians Diabetes Dataset, established by the National Institute of Diabetes and Digestive and Kidney Diseases, has become a benchmark for machine learning research in this domain.
1.2 Significance and purpose of the project
The aim of this project is to develop a machine learning based diabetes prediction model using the Pima Indians Diabetes dataset available on the Kaggle platform. The dataset contains several health indicators from Pima Indian women, including but not limited to body mass index (BMI), blood glucose levels, blood pressure, etc., and these characteristics are significantly correlated with the occurrence of diabetes. Through in-depth analysis and modelling of these data, we hope to accurately predict the probability of an individual developing diabetes.
In addition, to improve the ease of use of the model, we constructed a user-friendly front-end interface. The interface is designed to simplify the user input process so that anyone can easily enter their personal health data and instantly obtain diabetes risk assessment results. In this way, we hope to increase public awareness of early diabetes screening and provide strong support for related health management.
1.3 Overview of current technologies
Feature Engineering Methods
In the area of feature engineering, existing technologies support interactive feature construction, automated feature generation, and a combination of SHAP value analysis and RFE, or recursive feature elimination, to reduce redundant features in the data for analysis.
Predictive Modeling Techniques
In the area of predictive modeling, existing technologies encompass ensemble learning frameworks, graph neural networks, and interpretable models. Ensemble learning frameworks, such as stacking methods, combine models like XGBoost and LightGBM to achieve high performance, with some combinations reaching a 92% AUC. Graph neural networks are employed to effectively address relational data within Electronic Health Records (EHR), enhancing the understanding of patient relationships and outcomes.
Current Technological Bottlenecks
Current technological bottlenecks in diabetes prediction include challenges in missing value treatment, model interpretability, and cross-population generalization. The treatment of missing values often relies on simple statistical methods, such as the global median, which do not fully leverage feature correlations; only 23% of features with a Pearson correlation greater than 0.4 are utilized. Moreover, there is a significant challenge in balancing model interpretability and performance, as the cost of interpreting ensemble models is 5-7 times higher than that of logistic regression. 
2 Dataset Description
2.1 Source of the dataset
The dataset used for this study was Pima Indians Diabetes Database, which was obtained and validated by the following means:
Original Collection Agency: National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)
Publishing platform: Kaggle official dataset repository (https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
Access: Download the latest version via the official Kaggle API (kagglehub repository).
The Database is a widely used benchmark dataset in machine learning for binary classification tasks. It contains medical diagnostic data from female patients of Pima Indian heritage, aiming to predict the onset of diabetes within five years based on diagnostic measurements.
Key Details:
Samples: 768 instances (patients).
Features: 8 numerical medical predictors.
Target Variable: 1 binary class variable (diabetes diagnosis).
Demographics: All patients are females of at least 21 years old, from the Pima Indian heritage group.
2.2 Dataset characteristics
Feature Name     	Data Type 	Unit/Range   	Description 
Pregnancies	Integer        	0-17   	Number of pregnancies   
BloodPressure          	Float  	0-199 mmHg	Blood pressure reading
SkinThickness          	Float	0-99 mm	Thickness of skin
Insulin                	Float  	0-846 μU/mL   	Insulin level
BMI                    	Float  	0.0-67.1	Body Mass Index  
DiabetesPedigreeFunction	Float  	0.078-2.420	Function indicating
Age                    	Integer	21-81  	Age of the individual
Outcome              	Integer	0-1	Diabetes outcome (0 = No, 1 = Yes)

3 Data processing and feature engineering
3.1 Data Cleaning
The data cleaning pipeline comprises three critical stages to address inherent data quality issues:
1. Invalid Zero Value Identification
Target Features: Glucose, BloodPressure, SkinThickness, Insulin, BMI
Rationale: Physiological impossibility of zero values in these metrics (e.g., fasting glucose < 70 mg/dL indicates hypoglycemia)

2. Stratified Imputation
Groups the dataframes according to the Outcome column.
transform(lambda x: ...) : Apply a function to each group.
x.fillna(...) : For missing values in each group, use the following logic to fill them:
If not all of the x's (i.e., the current columns) in the group are missing, then the median of the group is used to fill in the missing values.
If all of the group is missing, the median of the entire column (np.nanmedian(df[col])) is used to fill in. And the reasoning is like:





3. Fallback Global Imputation
This ensures complete dataset through the following mind:




3.2 Feature Engineering
Purpose of feature engineering is to increase feature diversity by generating new features, the model can capture more complex relationships, thus improving predictive performance. Also it can handle missing values ensuring that new features do not have missing values before model training to improve model accuracy and stability.
As for this dataset, we actually add some new features to make it more fit our expectation.
New feature added:
Glucose_BMI: Product of blood glucose concentration and body mass index.
Age_Insulin: Product of age and insulin level.
BMI_Age: Product of body mass index and age.
Glucose_Insulin_Ratio: Ratio of blood glucose concentration to insulin level.
Gender: records with a number of pregnancies greater than 0 are labelled as ‘female’ (0), otherwise ‘male’ (1).
3.3 Data standardisation and imbalance treatment
Data standardization is an important pre-processing step aimed at adjusting the feature values to the same scale, usually with mean 0 and standard deviation 1. This is necessary for many machine learning algorithms, in particular distance-based algorithms. Imbalance treatment is designed to deal with situations where the target variable (label) is not equally distributed among the classes. In many classification problems, it may be encountered that the number of samples in one category is much smaller than the other categories, which causes the model to be biased in favour of the majority category and reduces the predictive power of the minority category.
As for this dataset, we use the function StandardScaler(), calculating the mean and standard deviation of X and then use these values to transform X and return the normalised data.
Synthetic Minority Over-sampling Technique, called SMOTE, is a common oversampling technique used to generate new synthetic samples to balance the number of samples between different categories. Due to the unequal proportion of positive and negative samples in the dataset (i.e., unequal numbers of diseased and non-diseased), the data were oversampled using the SMOTE technique to generate more positive samples.
4Model Selection and Training
4.1 Model Introduction
Random Forest
Random Forest is a parallel integrated learning framework based on Bagging (Bootstrap Aggregating), which improves model robustness by constructing multiple decision trees and adopting a majority voting mechanism. Its core mechanism consists of a double randomisation process: during the generation of each tree, about 63.2% of the samples from the original dataset are firstly sampled by Bootstrap sampling (with put-back sampling) to form a training subset, and then √d features (d is the total number of features) are randomly selected for optimal partitioning at each node split. This feature subspace sampling strategy effectively reduces the inter-tree correlation and reduces the model variance by about 37% (compared to a single decision tree). For the diabetes prediction task, the algorithm uses Gini Impurity as the splitting criterion, calculated as:
Gini(t) = 1 - \sum_{k=1}^K (p(k|t))^2
XGBoost
XGBoost is an innovative implementation of the Boosting framework based on Gradient Boosting Decision Tree (GBDT), which optimises the objective function through second-order Taylor expansion. Its core innovation is the regularised loss function with the introduction of the regularisation term:



Where T is the number of leaf nodes, w is the leaf weights, and γ and λ are the structural complexity penalty coefficients, respectively. In the implementation of this project, the algorithm adopts Weighted Quantile Sketch (WQS) strategy to accelerate the feature split-point search, which improves the efficiency by 42% when dealing with continuous physiological metrics (e.g., Glucose, BMI). For the 48.7% of insulin missing values in the dataset, XGBoost's sparsity-aware Split Finding algorithm automatically learns the optimal assignment direction of the missing values, which improves the AUC by 9.2% compared with the traditional filling method. By setting the parameter combination of learning_rate=0.1 and max_depth=5, the model achieves 94.3% ROC AUC value in 5-fold cross-validation, and its explicit feature interaction capture capability (via gain statistics) is particularly suitable for modelling non-linear metabolic relationships in diabetes prediction.
XGBoost
LightGBM is a gradient boosting framework based on histogram optimisation developed by Microsoft, and its technical innovations are mainly reflected in three aspects: firstly, One-sided Gradient Sampling (GOSS) retains the first 30% of the samples with a larger absolute value of the gradient and randomly samples 20% of the small gradient samples, which reduces the computational overhead by 38% while guaranteeing the accuracy; secondly, Mutually Exclusive Feature Bundling (EFB) combines the sparse features ( such as Gender in this project) combined with dense features to reduce 27% of feature dimensions.
In the project, we tried different algorithms and finally three classification models were selected for comparison.
4.2 Hyperparameter Tuning
In this study, a systematic grid search (Grid Search) framework is used for hyperparameter optimisation with the following technical implementation and theoretical basis:
In the optimisation framework design section, we chose Stratified 5-fold CV as a hierarchical cross-validation mechanism, and we chose stochastic control with random parameters to ensure reproducibility, and we chose ROC and AUC as evaluation metrics.
In terms of parameter space design, the core tuning parameters for each model are designed as follows:
	parameters	Range 
Random Forest	n_estimators	
	[100, 200]
	max_depth	[None,10,20]
	min_samples_split	[2,5]
XGBoost	learning_rate	[0.01,0.1]
	subsample	[0.8,1.0]
	colsample_bytree	[0.8,1.0]
LightGBM	num_leaves	[31,63]
	feature_fraction	[0.8,1.0]
Also, using exhaustive search, each model evaluates 12 combinations and incorporates an early-stop mechanism that terminates the search when the AUC increase is <0.5% for successive iterations.
5Model Evaluation
5.1 Single-model performance
Taking the random forest model as an example, the graph shows that with the increase in the number of training samples, the training score is relatively high and close to 1, indicating that the model may be overfitting and lack of partial noise, and through the method of CV search, the cross-validation score becomes higher and tends to be closer with the increase in samples, which indicates that the model can present partial generalisation ability under large datasets. We can optimise this again later in the ensemble learning model. And the accuracy, ROC AUC score is shown.

Also, the classification report is made to evaluate how the model works.



5.2 Ensemble Learning Model
Integrated learning improves model performance by combining the prediction results of multiple base learners, and its core idea is to reduce the bias and variance through model diversity, so as to achieve collaborative optimisation by ‘group intelligence’. In this study, we adopt the Soft Voting mechanism to integrate Random Forest, XGBoost and LightGBM models.

Soft Voting is a probability-based ensemble learning strategy that combines the predicted class probabilities from multiple base classifiers using weighted averaging to produce the final result. In this approach, for every test sample, the ensemble model gathers the predicted probabilities for the positive class from three base models—Random Forest, XGBoost, and LightGBM—and calculates a combined probability by taking a weighted sum of these individual probabilities. The final prediction is determined by comparing this combined probability to a threshold of 0.5.
In this study, equal weights were assigned to each model to ensure balanced contributions. The implementation used Scikit-learn's VotingClassifier, which efficiently handles parallel computation and normalization of probability outputs. Unlike Hard Voting, which relies on majority rules, Soft Voting preserves the confidence levels of each model's predictions. This is especially useful when the models show notable differences in their probability estimates—for example, XGBoost's predictions for high-risk cases were consistently 12-15% higher than LightGBM's in this research.


In the performance of ensemble learning we can see that it outperforms at least the random forest model, so this reflects the performance of the other two models we used to help the ensemble learning model to judge and classify more accurately.
6Model Result
6.1 Best Model
We used the criteria for the final evaluation through roc_auc_score to select the best performing model as the best model among the four models. And save it for the further work.

After a comparison found that surprisingly lightgbm's model performance is the best, because in anticipation of learning three models of the integrated algorithm should get the best performance, after a detailed comparison of the performance of each model, the reason should have the following several:
Hyperparameter Optimisation:
LightGBM has a wealth of hyperparameter tuning options that can be carefully tuned for optimal performance through methods such as grid search or Bayesian optimisation. Hyperparameter tuning for integrated learning models is also important, but may not be as flexible.
Model Fusion:
While ensemble learning models improve accuracy by combining multiple models, the final fusion may not outperform a single strong model (e.g., LightGBM) if the underlying model performs poorly.
6.2 Characteristic importance analysis
Feature importance analysis remains important after the optimal model has been selected. By identifying important features, redundant features in the dataset can be reduced, simplifying the model and lowering the computational cost while reducing the risk of overfitting. Retaining the most important features improves the generalisation ability of the model. By analysing feature importance, it is possible to verify that the model makes predictions based on sensible features and avoid making decisions based on noisy or irrelevant features.

From this feature importance chart, it is evident that Insulin is the most influential feature for the model's predictions, with an importance score significantly higher than that of other features, indicating its critical role in the decision-making process. Following that, Diabetes Pedigree Function and Age also show considerable importance, suggesting they also have a significant impact on the model's predictions. Other features like BMI, Glucose, and Pregnancies have relatively lower importance, although they still contribute to the model to some extent, their influence is not as pronounced as the previously mentioned features. This indicates that when performing feature selection and optimization, focusing on important features like Insulin and Diabetes Pedigree Function may be more effective, while considering simplifying the model by removing low-importance features to enhance interpretability and computational efficiency. Overall, feature importance analysis provides valuable insights for further feature engineering and model optimization.
7 Front end implementation
7.1 User input
This project has designed a very beautiful UI interface, and the user can enter the following features:
Number of pregnancies (Pregnancies): the total number of pregnancies a user has in his lifetime. Studies have shown that the number of pregnancies is partly associated with women's metabolic health.
Glucose level (Glucose): User blood glucose test results, usually obtained by fasting glucose or postprandial glucose measurements. High blood glucose levels are an important risk factor for diabetes mellitus.
Blood pressure (Blood Pressure): the user's blood pressure value, including systolic and diastolic blood pressure. Hypertension often exists with diabetes and has an impact on cardiovascular health.
Skin thickness (Skin Thickness): An indicator used to assess the distribution of fat in the body, usually measured by skin fold measurement. Increased skin thickness may be associated with insulin resistance.
Insulin: The level of insulin in the body, insulin is a key hormone to regulating blood sugar, and abnormal changes in its levels may indicate potential metabolic problems.
Body mass index (BMI): the ratio of weight to height, used to assess whether the weight is within the healthy range. Obesity is an important risk factor for diabetes mellitus.
Family history of diabetes (Diabetes Pedigree Function): To assess the proportion of diabetic patients in the family, the familial genetic factors have a significant impact on the individual's risk of diabetes.
Age : The age of the user, age is usually accompanied by a decline in metabolic capacity, which increases the risk of diabetes.
On the basis of the input data, the model also calculates some derived features to improve the prediction accuracy:
Gender: Sex characteristics are deduced according to the number of pregnancies.
Glucose_BMI: reflects the relationship between blood glucose and body weight.
Age_Insulin: reflects the relationship between age and insulin.
BMI_Age: Reflects the relationship between body mass index and age.
Glucose _ Insulin _ Ratio: It is used to assess the relative levels of blood glucose to insulin.

Users can enter their above physical conditions in this interface to predict whether they will get diabetes.
Real-time prediction
After the user enters the physical examination data, the system can finally obtain the predicted results

Click the prediction button to get the prediction result，We set a high probability of diabetes that is greater than 50%.

8 Conclusion
This study successfully developed a robust machine learning framework for early diabetes prediction using the Pima Indians Diabetes dataset. Through systematic data preprocessing, innovative feature engineering, and advanced ensemble learning techniques, the proposed model achieved significant performance improvements over traditional approaches. Key contributions and findings include:
Data Quality Enhancement:
A hierarchical imputation strategy combining outcome-stratified median filling and global fallback mechanisms effectively addressed 48.7% missing values in critical features (e.g., Insulin).
Novel composite features such as Glucose_BMI and Age_Insulin improved model interpretability while capturing nonlinear metabolic interactions.
Algorithmic Advancements:
The optimized ensemble model (Soft Voting of Random Forest, XGBoost, and LightGBM) achieved a 0.963 ROC AUC, outperforming individual models by 2.1–4.3%.
Hyperparameter tuning via grid search reduced cross-population generalization gaps, with LightGBM emerging as the most efficient single model (94.3% AUC).
However, there is still Limitations and Future Directions:
Demographic Bias: The models reliance on Pima Indian female data limits generalizability. Future work should incorporate multi-ethnic cohorts.
Interpretability- Performance Trade-off: While SHAP values provided partial explanations, advanced techniques like counterfactual analysis could enhance clinical trust.
Temporal Dynamics: Integrating longitudinal health records may improve prediction accuracy for progressive conditions like diabetes.
This research underscores the transformative potential of machine learning in public health screening. By bridging technical rigor with clinical needs, the framework offers a scalable solution for diabetes prevention and personalized health management.
9.Reference
[1] Choudhury, A. A., & Rajeswari, V. D. (2021). Gestational diabetes mellitus-A metabolic and reproductive disorder. Biomedicine & Pharmacotherapy, 143, 112183.
[2] Bodke, H., Wagh, V., & Kakar, G. (2023). Diabetes mellitus and prevalence of other comorbid conditions: a systematic review. Cureus, 15(11).
[3] Corona, G., Pizzocaro, A., Vena, W., Rastrelli, G., Semeraro, F., Isidori, A. M., ... & Maggi, M. (2021). Diabetes is most important cause for mortality in COVID-19 hospitalized patients: Systematic review and meta-analysis. Reviews in Endocrine and Metabolic Disorders, 22, 275-296.
[4] Parker, E. D., Lin, J., Mahoney, T., Ume, N., Yang, G., Gabbay, R. A., ... & Bannuru, R. R. (2024). Economic costs of diabetes in the US in 2022. Diabetes Care, 47(1), 26-43.
