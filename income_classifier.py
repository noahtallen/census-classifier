"""
Noah Allen and Rachel Leone
Machine Learning Project 3 

Thanks to this Official Tensorflow example for helping us learn how to import
the dataset: https://github.com/tensorflow/models/tree/58deb0599f10dc5b33570103339fb7fa5bb876c3/official/wide_deep
"""

import tensorflow as tf
import os

# See here: https://github.com/tensorflow/models/blob/58deb0599f10dc5b33570103339fb7fa5bb876c3/official/wide_deep/census_dataset.py#L41
_CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
]

_CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                        [0], [0], [0], [''], ['']]

_BATCH_SIZE = 500

# See here: https://github.com/tensorflow/models/blob/58deb0599f10dc5b33570103339fb7fa5bb876c3/official/wide_deep/census_dataset.py#L89
def get_columns(do_linear):
    """Builds a set of feature columns."""
    
    # Continuous variable columns
    age = tf.feature_column.numeric_column('age')
    # education_num = tf.feature_column.numeric_column('education_num')
    capital_gain = tf.feature_column.numeric_column('capital_gain')
    capital_loss = tf.feature_column.numeric_column('capital_loss')
    hours_per_week = tf.feature_column.numeric_column('hours_per_week')

    # Categorical variable columns
    education = tf.feature_column.categorical_column_with_vocabulary_list(
        'education', [
            'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
            'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
            '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])

    marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
        'marital_status', [
            'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
            'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])

    # relationship = tf.feature_column.categorical_column_with_vocabulary_list(
    #     'relationship', [
    #         'Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried',
    #         'Other-relative'])

    workclass = tf.feature_column.categorical_column_with_vocabulary_list(
        'workclass', [
            'Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
            'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'])
    
    native_country = tf.feature_column.categorical_column_with_vocabulary_list('native_country', [
        'United-States', 'Cambodia, England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)',
        'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland',
        'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador',
        'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia',
        'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands'
    ])

    occupation = tf.feature_column.categorical_column_with_vocabulary_list('occupation', [
        'Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty',
        'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving',
        'Priv-house-serv', 'Protective-serv', 'Armed-Forces'
    ])

    gender = tf.feature_column.categorical_column_with_vocabulary_list('gender', [
        'Female', 'Male'
    ])

    # Note: we do not use this by itself, only as a crossed feature:
    race = tf.feature_column.categorical_column_with_vocabulary_list(
        'race', ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
    )

    # Synthetic Features:
    # Bucketized age: (helps us ignore outliers at very old or young ages)
    age_buckets = tf.feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    
    # Three buckets: one for under time, one for full time, and one for over time
    hours_per_week_bucket = tf.feature_column.bucketized_column(hours_per_week, boundaries=[39, 41])

    # Cross race with education:
    race_with_education = tf.feature_column.crossed_column([race, education], hash_bucket_size=1000)
    # education_with_occupation = tf.feature_column.crossed_column([education, occupation], hash_bucket_size=1000),

    synthetic_features = [ 
        age_buckets, 
        hours_per_week_bucket, 
    ]

    neural_net_features = [
        tf.feature_column.indicator_column(education),
        tf.feature_column.indicator_column(marital_status),
        tf.feature_column.indicator_column(workclass),
        tf.feature_column.indicator_column(race_with_education),
        # tf.feature_column.indicator_column(education_with_occupation),
        tf.feature_column.indicator_column(native_country),
        tf.feature_column.indicator_column(occupation),
        tf.feature_column.indicator_column(gender)
    ] + [capital_gain, capital_loss] + synthetic_features

    linear_classifier_features = [
        education, occupation, gender, race_with_education,
        marital_status, workclass, native_country,
        capital_gain, capital_loss
    ] + synthetic_features

    if do_linear: 
        return linear_classifier_features

    return neural_net_features

def train_importer():
    """Imports the training data set"""
    return importer('adult-data.csv', True)

def test_importer():
    """Imports the test data set"""
    return importer('adult-data-test.csv', False)

# Reads the csv file and returns it as a tf dataset
def importer(filename, should_shuffle):
    """Imports data set from the given file"""
    def parse_line( line ):
        fields = tf.decode_csv(line, _CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, fields))
        # Clean up the data:
        labels = tf.strings.strip(features.pop('income_bracket'))
        classes = tf.equal(labels, '>50K')  # For binary classification
        return features, classes

    dataset = tf.data.TextLineDataset(os.path.abspath(filename))
    # dataset = tf.data.experimental.CsvDataset(record_file, record_defaults=_CSV_COLUMN_DEFAULTS)
    dataset = dataset.map(parse_line)
    if should_shuffle:
        dataset = dataset.shuffle(1000)
    return dataset.batch(_BATCH_SIZE)

def run_classifier():
    """Trains and evaluates two classifiers on the census data"""
    do_linear = True
    if do_linear:
        # Linear Classifier:
        estimator = tf.estimator.LinearClassifier(feature_columns=get_columns(do_linear))
    else:
        # Neural Net:
        estimator = tf.estimator.DNNClassifier(feature_columns=get_columns(do_linear), hidden_units=[10, 10, 10, 10, 10], n_classes=2)

    estimator.train(input_fn=train_importer)
    results = estimator.evaluate(input_fn=test_importer)
    print(results)

run_classifier()