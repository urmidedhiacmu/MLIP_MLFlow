import mlflow
import numpy as np

# TODO: Set tht MLFlow server uri
uri = ___________________
mlflow.set_tracking_uri(uri=uri)

# TODO: Provide model path/url
logged_model = ___________________

# Load model as a PyFuncModel.
loaded_model = mlflow.sklearn.load_model(logged_model)

# Input a random datapoint
np.random.seed(42)
data = np.random.rand(1, 64)

# TODO: Predict the output for the data. You might need to use a pandas DataFrame due to a constraint from MLFlow.
prediction = loaded_model.predict(___________)

# Print out prediction result
print(prediction)
