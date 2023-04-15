import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('flight_data.csv')

# Preprocess the data
df = df.drop(['flight_number', 'scheduled_departure', 'scheduled_arrival'], axis=1)
df['departure_delayed'] = (df['departure_delay'] > 0).astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('departure_delayed', axis=1), df['departure_delayed'], test_size=0.2)

# Train the logistic regression model
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
