using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using System;
using System.IO;
using TaxiFarePrediction.Models;

namespace TaxiFarePrediction
{
    class Program
    {
        private static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-train.csv");
        private static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-test.csv");
        private static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");
        private static TextLoader _textLoader;

        static void Main()
        {
            var mlContext = new MLContext(seed: 0);

            _textLoader = mlContext.Data.CreateTextReader(new TextLoader.Arguments
            {
                Separator = ",",
                HasHeader = true,
                Column = new[]
                {
                    new TextLoader.Column("VendorId", DataKind.Text, 0),
                    new TextLoader.Column("RateCode", DataKind.Text, 1),
                    new TextLoader.Column("PassengerCount", DataKind.R4, 2),
                    new TextLoader.Column("TripTime", DataKind.R4, 3),
                    new TextLoader.Column("TripDistance", DataKind.R4, 4),
                    new TextLoader.Column("PaymentType", DataKind.Text, 5),
                    new TextLoader.Column("FareAmount", DataKind.R4, 6)
                }
            });

            Console.WriteLine(">>>> Building and Training model...");
            Console.WriteLine();

            var model = BuildAndTrain(mlContext, _trainDataPath);

            Console.WriteLine(">>>> Saving model...");
            Console.WriteLine();

            SaveModelAsFile(mlContext, model, _modelPath);

            Console.WriteLine(">>>> Evaluating...");
            Console.WriteLine();

            var metrics = Evaluate(mlContext, model, _testDataPath);

            Console.WriteLine();
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Model quality metrics evaluation         ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       R2 Score:      {metrics.RSquared:0.##}");
            Console.WriteLine($"*       RMS loss:      {metrics.Rms:#.##}");
            Console.WriteLine();


            Console.WriteLine(">>>> Testing single prediction...");
            Console.WriteLine();

            var taxiTrip = new TaxiTrip
            {
                VendorId = "VTS",
                RateCode = "1",
                PassengerCount = 1,
                TripTime = 1140,
                TripDistance = 3.75f,
                PaymentType = "CRD",
                FareAmount = 0 // To predict. Actual/Observed = 15.5
            };

            var prediction = Predict(mlContext, _modelPath, taxiTrip);

            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted fare: {prediction.FareAmount:0.####}, actual fare: 15,5");
            Console.WriteLine($"**********************************************************************");
            Console.ReadKey();
        }

        public static ITransformer BuildAndTrain(MLContext mlContext, string dataPath)
        {
            var dataView = _textLoader.Read(dataPath);

            var pipeline = mlContext.Transforms.CopyColumns("FareAmount", "Label")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("VendorId"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("RateCode"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("PaymentType"))
                .Append(mlContext.Transforms.Concatenate(
                    "Features",
                    "VendorId",
                    "RateCode",
                    "PassengerCount",
                    "TripTime",
                    "TripDistance",
                    "PaymentType"))
                .Append(mlContext.Regression.Trainers.FastTree());

            return pipeline.Fit(dataView);
        }

        private static void SaveModelAsFile(MLContext mlContext, ITransformer model, string modelPath)
        {
            using (var fileStream = new FileStream(modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
                mlContext.Model.Save(model, fileStream);
        }

        private static RegressionMetrics Evaluate(MLContext mlContext, ITransformer model, string testDataPath)
        {
            var dataView = _textLoader.Read(testDataPath);
            var predictions = model.Transform(dataView);
            return mlContext.Regression.Evaluate(predictions, "Label", "Score");
        }

        private static TaxiTripFarePrediction Predict(MLContext mlContext, string modelPath, TaxiTrip taxiTrip)
        {
            ITransformer loadedModel;
            using (var stream = new FileStream(modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
                loadedModel = mlContext.Model.Load(stream);

            var predictionFunction = loadedModel.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(mlContext);
            return predictionFunction.Predict(taxiTrip);
        }
    }
}
