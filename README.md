# Resampling for Classification vs. Class Probability Prediction

Oversampling and undersampling techniques are often used in machine learning classification tasks on imbalanced data. The primary motivation is for refined calibration and performance. Many times classification accuracy alone will over-state the quality of a particular forecast when minority class occurrence is very rare (< 1%). A model might achieve 99% accuracy but incorrectly classify every single minority outcome simply because 99% of observations are in the majority class. False positive and false negative rates, also called Type I and II errors, provide a more nuanced view of predictive accuracy. Resampling methods help to adjust for imbalanced outcomes by artificially changing the proportion of classes in data to improve minority class performance. The confusion matrix along with specificity and sensitivity rates are then used to monitor classification improvements across each class.

However in financial cash flow analysis the model output is typically predicted class probability multiplied by dollar-denominated principal balance, not class ID:
* **Default probability** as credit risk adjustment or % balance outflow
* **Prepayment probability** as % balance outflow
* **Attrition probability** as % line and balance outflow 

In a classification task, the class probability is converted to class ID using the argmax function, where each observation is assigned to the class with the highest predicted probability. For cash flow adjustments the class probability is used directly without rounding to 0 or 1. Individual loans aren’t assigned to binary status of defaulted/non-defaulted, but instead a percentage of their balance is subtracted each month based on predicted default probability. Performance isn’t measured by classification accuracy or the confusion matrix but by dollar-denominated error in aggregate balance adjustment, i.e. error in total predicted dollar losses.
