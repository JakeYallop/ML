using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace ER_Inbound_Calls_Prediction
{
    public static class MatrixUtils
    {
        public static Matrix<double> StackSelf(this Matrix<double> m, int n)
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
        public static void LogPrediction(double pred, double error) => Console.WriteLine("Prediction: {0} Error: {1}", pred.ToString("N8"), error.ToString("N8"));
        public static double Relu(double input) => input > 0 ? input : 0;
        public static double Relu2Deriv(double input) => input > 0 ? 1 : 0;

    
        public static MatrixBuilder<double> M = Matrix.Build;
        public static VectorBuilder<double> V = Vector.Build;

        public const double Alpha = 0.1;
        public const int HiddenNodes = 8;
        public const int OutputNodes = 1;
        public const int InputNodes = 3;

        public static Matrix<double> TestData = M.DenseOfRowArrays(new double[] { 1, 0, 1 }, new double[] { 0, 1, 1 }, new double[] { 1, 1, 1 }, new double[] { 0, 0, 1 });
        public static Vector<double> Correct = V.DenseOfArray(new double[] { 1, 1, 0, 0 });

        public const int Iterations = 500;
        static void Main(string[] args)
        {
            Matrix<double> w0 = M.Random(InputNodes, HiddenNodes, new ContinuousUniform());
            Matrix<double> w1 = M.Random(HiddenNodes, OutputNodes, new ContinuousUniform());

            int iterCount = 0;

            for (int i = 0; i < Iterations; i++)
            {
                iterCount++;
                for (int j = 0; j < TestData.RowCount; j++)
                {
                    //Forwards Propagation
                    Vector<double> l0 = TestData.Row(j);
                    Vector<double> l1 = w0.LeftMultiply(l0).Map(Relu);
                    Vector<double> l2 = w1.LeftMultiply(l1);
                    
                    double goal = Correct[j];
                    double prediction = l2.Sum();
                    double error = Math.Pow(goal - prediction, 2);
                    LogPrediction(prediction, error);
                    
                    //Backward Propagation
                    //Calculate Delta at each layer
                    Vector<double> l2Delta = V.Dense(1, goal - prediction);
                    Vector<double> l1Delta = w1.Transpose().LeftMultiply(l2Delta).Map2((a, b) => a * Relu2Deriv(b), l1);
                    //Update Weights
                    w0 = w0.Add(l1Delta.ToRowMatrix().StackSelf(InputNodes).PointwiseMultiply(l0.ToRowMatrix().StackSelf(HiddenNodes).Transpose()).Multiply(Alpha));
                    w1 = w1.Add(l1.Multiply(l2Delta.Sum()).Multiply(Alpha).ToColumnMatrix());
                }
            }
            Console.ReadLine();
        }
    }
}
