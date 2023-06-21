# NTCIR'17 Social Media Subtask
## Adverse Drug Event detection for social media texts in Japanese, English, German, and French

The following measure are used, mostly based ob the scikit-learn library:


1) **Binary**: Calculates the performance of classifying a document into the classes "contains ADE" (positive) vs. "does not contain ADE" (negative). A document is considered to contain an ADE if a least one symptom (class) is positive (1). The most interesting scores in this case are precision, recall and F1 for the *positive* class.

2) **Per class**: Calculates precision, recall and F1 score for each class (symptom). This is useful to see if there are any differences in how systems detect different symptoms (individual scores per class).

3) **(Full) per label**: Calculates precision, recall and F1 score for each *label* (0 and 1) across samples and classes.

4) **Exact match accuracy**: Calculates the percentage of exact matches across all samples. The system has to predict a perfect labeling of a sample; as soon as one symptom is not correctly predicted, the sample will not be counted.



The predictions file is expected to have the exact same structure as the train data file.
Install the necessary libraries and run the script like so:

`pip install -r requirements.txt`

`python evaluate_ntcir_2023.py -gold_file gold.csv -prediction_file pred.csv`


