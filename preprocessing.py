from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocessing_pipeline(df, test_size=0.3, random_state=42):

    X = df.drop(columns=['y'])
    y = df['y']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    encoders = {}
    le = LabelEncoder()
    for feature in ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']:
        X_train[feature] = le.fit_transform(X_train[feature])
        X_test[feature] = le.transform(X_test[feature])
        encoders[feature] = le

    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    encoders['y'] = le

    scalers = {}
    for feature in ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']:
        scaler = StandardScaler()
        X_train[feature] = scaler.fit_transform(X_train[[feature]])
        X_test[feature] = scaler.transform(X_test[[feature]])
        scalers[feature] = scaler

    smote_tomek = SMOTETomek(random_state=random_state)
    X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)

    return X_resampled, X_test, y_resampled, y_test, encoders, scalers

def cv_5fold():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def cv_10fold():
    return StratifiedKFold(n_splits=10, shuffle=True, random_state=42)