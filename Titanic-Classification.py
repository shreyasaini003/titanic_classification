import pandas as pd
import numpy as np
import tensorflow as tf

# Load the training and test datasets
train_data = pd.read_csv('train_dataset.csv')
test_data = pd.read_csv('test_dataset.csv')

# Preprocess the data
train_data = train_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
test_data = test_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)

train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})
train_data['Fare'].fillna(train_data['Fare'].mean(), inplace=True)

test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)
test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})
test_data['Fare'].fillna(test_data['Fare'].mean(), inplace=True)

# Split the data into features and target
X_train = train_data.drop('Survived', axis=1).values
y_train = train_data['Survived'].values

X_test = test_data.drop('Survived', axis=1).values
y_test = test_data['Survived'].values

# Create a neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(6,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=64)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

# Make predictions on new data
new_data = np.array([[3, 0, 30, 0, 0, 7.75]])  # Example data
new_data[:, 1] = np.where(new_data[:, 1] == 'male', 0, 1)  # Map gender to numerical values
new_data[:, 5] = np.where(np.isnan(new_data[:, 5]), np.nanmean(new_data[:, 5]), new_data[:, 5])  # Fill missing fare
# value with mean
prediction = model.predict(new_data)
print('Survival prediction:', np.round(prediction))
