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

_BATCH_SIZE = 1000

# See here: https://github.com/tensorflow/models/blob/58deb0599f10dc5b33570103339fb7fa5bb876c3/official/wide_deep/census_dataset.py#L89
def get_columns():
    """Builds a set of wide and deep feature columns."""
    # Continuous variable columns
    age = tf.feature_column.numeric_column('age')
    education_num = tf.feature_column.numeric_column('education_num')
    capital_gain = tf.feature_column.numeric_column('capital_gain')
    capital_loss = tf.feature_column.numeric_column('capital_loss')
    hours_per_week = tf.feature_column.numeric_column('hours_per_week')
    education = tf.feature_column.categorical_column_with_vocabulary_list(
        'education', [
            'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
            'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
            '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])

    marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
        'marital_status', [
            'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
            'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])

    relationship = tf.feature_column.categorical_column_with_vocabulary_list(
        'relationship', [
            'Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried',
            'Other-relative'])

    workclass = tf.feature_column.categorical_column_with_vocabulary_list(
        'workclass', [
            'Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
            'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'])
    
    neural_net_features = [
        tf.feature_column.indicator_column(education),
        tf.feature_column.indicator_column(marital_status),
        tf.feature_column.indicator_column(relationship),
        tf.feature_column.indicator_column(workclass),
    ] + [age, education_num, capital_gain, capital_loss, hours_per_week]

    linear_classifier_features = [
        age, education_num, capital_gain, capital_loss, hours_per_week, education,
        marital_status, relationship, workclass
    ]

    return linear_classifier_features
    # return neural_net_features + 

def train_importer():
    return importer('adult-data.csv')

def test_importer():
    return importer('adult-data-test.csv')

# Reads the csv file and returns it as a tf dataset
def importer(filename):
    def format_data( age, workclass, fnlwgt, education, education_num,
        marital_status, occupation, relationship, race, gender,
        capital_gain, capital_loss, hours_per_week, native_country,
        income_bracket ):
        features = dict(zip(_CSV_COLUMNS, [ age, workclass, fnlwgt, education, education_num,
            marital_status, occupation, relationship, race, gender,
            capital_gain, capital_loss, hours_per_week, native_country,
            income_bracket ]))
        labels = features.pop('income_bracket')
        classes = tf.equal(labels, '>50K')  # For binary classification
        return features, classes

    record_file = [ os.path.abspath(filename) ]
    dataset = tf.data.experimental.CsvDataset(record_file, record_defaults=_CSV_COLUMN_DEFAULTS)
    dataset = dataset.map(format_data)
    return dataset.shuffle(1000).batch(_BATCH_SIZE)

def run_classifier():
    # Linear:
    estimator = tf.estimator.LinearEstimator(feature_columns=get_columns())
    # Neural Net:
    # estimator = tf.estimator.DNNClassifier(feature_columns=get_columns(), hidden_units=[10, 10], n_classes=2)
    estimator.train(input_fn=train_importer)
    results = estimator.evaluate(input_fn=test_importer)
    print(results)

run_classifier()