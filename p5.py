# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Step 2: Load dataset
data = pd.read_csv("SAheart.csv")

# Step 3: Convert categorical to numeric
data['famhist'] = data['famhist'].map({'Present':1, 'Absent':0})

# Step 4: Features and label
X = data[['tobacco','ldl','adiposity','typea','obesity','alcohol','age','famhist']]
y = data['chd']

# Step 5: Normalize data
X = normalize(X)

# Step 6: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 7: Build ANN model
model = Sequential()
model.add(Dense(10, input_dim=8, activation='relu'))  # hidden layer
model.add(Dense(1, activation='sigmoid'))             # output layer

# Step 8: Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 9: Train model
model.fit(X_train, y_train, epochs=50, batch_size=16)

# Step 10: Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)

print("Accuracy:", accuracy)
