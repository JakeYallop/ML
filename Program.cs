using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System;

namespace ML_Relu
{
    public static class MatrixUtils
    {
        public static Matrix<double> StackMatrix(this Matrix<double> m, int n)
        {
            Matrix<double> final = m;
            for (int i = 0; i < n - 1; i++)
            {
                final = final.Stack(m);
            }
            return final;
        }

    }

    class Program
    {

        public static double ComputeError(double prediction, double goal) => Math.Pow(goal - prediction, 2);
        public static void LogPrediction(double prediction, double error) => Console.WriteLine("Error: " + error.ToString("N8") + " Prediction: " + prediction.ToString("N8"));
        public static double Relu(double input) => input > 0 ? input : 0;
        public static double ReluToDeriv(double input) => input > 0 ? 1 : 0;

        public static Matrix<double> weightsZeroToOne;
        public static Matrix<double> weightsOneToTwo;

        public static void Predict(double[] streetlight, double correctAnswer)
        {
            Vector<double> layerZero = Vector.Build.DenseOfArray(streetlight);
            Vector<double> layerOne = weightsZeroToOne.LeftMultiply(layerZero);
            layerOne.CoerceZero(a => a < 0);
            double pred = weightsOneToTwo.LeftMultiply(layerOne).Sum();

            Console.WriteLine("Predicted Value: " + pred.ToString("N6") + " Mean-Squared error: " + ComputeError(pred, correctAnswer).ToString("N6"));
        }

        static void Main(string[] args)
        {
            double prediction = 0;
            double goal = 0;
            double alpha = 0.2;
            int hiddenSize = 4;
            int numberOfInputs = 3;
            int iterations = 100;

            //Generate random weights
            weightsZeroToOne = DenseMatrix.Build.Random(3, hiddenSize, new ContinuousUniform()); //From layer 0 (3 inputs) to layer 1 (hiddenSize inputs)
            weightsOneToTwo = DenseMatrix.Build.Random(hiddenSize, 1, new ContinuousUniform()); //From layer 1 (hiddenSize inputs) to layer 2 (1 output)

            //Data (layer 0)
            Matrix<double> streetlightData = DenseMatrix.Build.DenseOfRowArrays(new double[][] {
              new double[] { 1, 0, 1 },
              new double[] { 0, 1, 1 },
              new double[] { 0, 0, 1 },
              new double[] { 1, 1, 1 },
            });

            //Correct Predictions
            double[] correctPredictions = { 1, 1, 0, 0 };

            Console.WriteLine("Running for {0} iterations", iterations);
            //List starting parameters
            Console.WriteLine("Starting Parameters: ");
            Console.WriteLine("weightsZeroToOne: ");
            Console.WriteLine(weightsZeroToOne);
            Console.WriteLine("WeightsOneToTwo: ");
            Console.WriteLine(weightsOneToTwo);
            
            for (int i = 0; i < iterations; i++)
            {
                for (int j = 0; j < streetlightData.RowCount; j++)
                {
                    goal = correctPredictions[j];
                    Vector<double> layerZero = streetlightData.Row(j);
                    Vector<double> layerOne = weightsZeroToOne.LeftMultiply(layerZero);
                    layerOne.CoerceZero(a => a < 0);

                    double layerTwo = weightsOneToTwo.LeftMultiply(layerOne).Sum();
                    double layerTwoDelta = (goal - layerTwo);

                    Matrix<double> layerOneDelta = weightsOneToTwo.Multiply(layerTwoDelta).Map2((a, b) => a * ReluToDeriv(b), Matrix.Build.DenseOfColumnVectors(layerOne));

                    //Create a 3x4 matrix from layer 1 (originally 1x4 -> stack rows 3 times)
                    Matrix<double> m = layerOneDelta.Transpose().StackMatrix(3);
                    //Create 3x4 matrix from layer 0 (originally 1x3 -> stack rows 4 times and rotate from 4x3 to 3x4)
                    Matrix<double> m2 = layerZero.ToRowMatrix().StackMatrix(4).Transpose();

                    //Update weighting
                    weightsZeroToOne = weightsZeroToOne.Add(m2.PointwiseMultiply(m).Multiply(alpha));
                    //weightsOneToTwo is a 4x1 matrix, so converting layerOne (4-Double vector) to a column matrix results in a 4x1 matrix (which can then be multiplied and added)
                    weightsOneToTwo = weightsOneToTwo.Add(layerOne.Multiply(layerTwoDelta).ToColumnMatrix() * alpha);

                    prediction = layerTwo;
                    LogPrediction(prediction, ComputeError(prediction, goal));
                }
            }
            Console.WriteLine("Training complete");
            Console.WriteLine("--------------------------------");
            Console.WriteLine();


            Predict(new double[] { 1, 0, 1 }, 1);
            Console.ReadLine();
        }
    }
}
