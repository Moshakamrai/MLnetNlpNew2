using System;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace TextClassificationApp
{
    class Program
    {
        static void Main(string[] args)
        {
            // 1. Create a new MLContext object to start ML.NET operations
            var mlContext = new MLContext();

            // 2. Load data from a text file (Make sure you have a data file called 'data.txt')
            var data = mlContext.Data.LoadFromTextFile<TextData>(@"C:\MainFolder\NewMlNetNLP\MLnetNlpNew\MLnetNlpNew\data.txt", separatorChar: ',', hasHeader: true);

            // 3. Split the data into training and test sets (80% training, 20% testing)
            var trainTestData = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

            // 4. Define the pipeline (transform text to numeric features)
            var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(TextData.Text))
    .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label"));





            // 5. Train the model using the training data
            var model = pipeline.Fit(trainTestData.TrainSet);

            // 6. Evaluate the model using the test data (optional, here we use classification metrics)
            var predictions = model.Transform(trainTestData.TestSet);
            var metrics = mlContext.BinaryClassification.Evaluate(predictions);
            Console.WriteLine($"Accuracy: {metrics.Accuracy}");

            // 7. Save the trained model to a file (sentiment_model.zip)
            mlContext.Model.Save(model, data.Schema, "sentiment_model.zip");
            Console.WriteLine("Model saved to sentiment_model.zip");

            // Step 3: Now let's load the saved model for real-time predictions

            // 8. Load the trained model
            ITransformer loadedModel = mlContext.Model.Load("sentiment_model.zip", out var modelInputSchema);
            Console.WriteLine("Model loaded successfully!");

            // 9. Use the loaded model for predictions (example for new data)

            // Create a prediction engine
            var predictionFunction = mlContext.Model.CreatePredictionEngine<TextData, Prediction>(loadedModel);

            // Test with different data

            // Test Case 1: Positive sentiment
            var testData1 = new TextData { Text = "I love programming!" };
            var prediction1 = predictionFunction.Predict(testData1);
            Console.WriteLine($"Test Case 1 - Input: '{testData1.Text}', Predicted label: {prediction1.PredictedLabel}");

            // Test Case 2: Negative sentiment
            var testData2 = new TextData { Text = "I hate bugs!" };
            var prediction2 = predictionFunction.Predict(testData2);
            Console.WriteLine($"Test Case 2 - Input: '{testData2.Text}', Predicted label: {prediction2.PredictedLabel}");

            // Test Case 3: Neutral sentiment (or could be positive based on model)
            var testData3 = new TextData { Text = "I like solving coding challenges." };
            var prediction3 = predictionFunction.Predict(testData3);
            Console.WriteLine($"Test Case 3 - Input: '{testData3.Text}', Predicted label: {prediction3.PredictedLabel}");

            // Test Case 4: Ambiguous sentiment
            var testData4 = new TextData { Text = "Debugging is fun but frustrating." };
            var prediction4 = predictionFunction.Predict(testData4);
            Console.WriteLine($"Test Case 4 - Input: '{testData4.Text}', Predicted label: {prediction4.PredictedLabel}");

            // Test Case 5: Very positive sentiment
            var testData5 = new TextData { Text = "I absolutely love coding and learning new technologies!" };
            var prediction5 = predictionFunction.Predict(testData5);
            Console.WriteLine($"Test Case 5 - Input: '{testData5.Text}', Predicted label: {prediction5.PredictedLabel}");
        }
    }

    // 10. Create a class for input data
    public class TextData
    {
        [LoadColumn(0)]  // This maps the first column to the Text property
        public string Text { get; set; }

        [LoadColumn(1)]  // This maps the second column to the Label property
        public bool Label { get; set; }  // Binary classification: 1 (True) or 0 (False)
    }

    // 11. Create a class for the prediction result
    public class Prediction
    {
        public bool PredictedLabel { get; set; }  // Predicted label will be Boolean as well
    }
}
