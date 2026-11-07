
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd


class Model:

    def __init__(self, dataset='Student_Performance.csv', test_size=0.2, random_state=42):

        # Store the dataset and configuration parameters
        self.dataset = dataset
        self.test_size = test_size
        self.random_state = random_state

        # Initialize the model and training status
        self.is_trained = False
        self.model = LinearRegression()

        # Load the dataset and select input features and target
        self.data = pd.read_csv(dataset)
        self.X = self.data[['Hours Studied', 'Previous Scores', 'Sleep Hours']]
        self.y = self.data['Performance Index']

    def split_data(self):

        # Split the dataset into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state
        )

    def train(self):

        # Train the linear regression model
        self.model.fit(self.X_train, self.y_train)
        self.is_trained = True

    def predict(self, study_hours, previous_scores, sleep_hours):

        # Prevent predictions if the model is not trained
        if not self.is_trained:
            return None

        # Create a DataFrame with the student's input data
        student = pd.DataFrame([[
            study_hours,
            previous_scores,
            sleep_hours
        ]], columns=self.X.columns)

        # Generate and return the predicted performance index
        prediction = self.model.predict(student)
        return prediction[0]

    def evaluate(self):

        # Predict results using the testing data
        predictions = self.model.predict(self.X_test)

        # Calculate model performance metrics
        mae = mean_absolute_error(self.y_test, predictions)
        r2 = r2_score(self.y_test, predictions)

        # Return evaluation metrics
        return {
            'MAE': mae,
            'R2': r2
        }
