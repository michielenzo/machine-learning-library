using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Numerics;
using Machine_Learning_Library.DeepLearning.NeuralNetwork;

namespace Machine_Learning_Library
{
    public class LineairRegression: IModel
    {

        private readonly List<Vector2> _dataset;
        public enum InterceptStrategies { GradientDescent, EquationSolving }
        public InterceptStrategies InterceptStrategy { get; set; }

        public double LearningRate = 0.1;
        public double MaxSteps = 100;
        public double MinStepSize = 0.01;
        public double MaxStepSize = 2;
        public double initialGuess = 0;

        public bool Logging;

        public LineairRegression()
        {
            _dataset = new List<Vector2>();
            InterceptStrategy = InterceptStrategies.EquationSolving;
            Logging = false;
        }

        public void Fit(double independent, double dependent)
        {
            _dataset.Add(new Vector2((float) independent, (float) dependent));
        }

        public double Predict(double dependent)
        {
            return CalculateIntercept() + CalculateSlope() * dependent;
        }

        public void Unfit() {
            _dataset.Clear();
        }

        public double GetIntercept()
        {
            return CalculateIntercept();
        }

        public double GetSlope()
        {
            return CalculateSlope();
        }

        public String FunctionAsString()
        {
            return "prediction = " + CalculateIntercept() + " + " + CalculateSlope() + " * dependent";
        }

        private double CalculateSlope()
        {
            double meanX = CalculateMeanX();
            double meanY = CalculateMeanY();

            double sumXyDistanceToMean = _dataset.Sum(point => (point.X - meanX) * (point.Y - meanY));
            double sumXDistanceToMean = _dataset.Sum(point => Math.Pow(point.X - meanX, 2));

            return sumXyDistanceToMean / sumXDistanceToMean;
        }

        private double CalculateIntercept()
        {
            switch (InterceptStrategy)
            {
                case InterceptStrategies.EquationSolving:
                    return -(CalculateSlope() * CalculateMeanX() - CalculateMeanY());
                case InterceptStrategies.GradientDescent:
                    double RegressionLine(double initialIntercept, double x) => initialIntercept + CalculateSlope() * x;
                    return GradientDescent(_dataset, RegressionLine);
                default:
                    return 0f;
            }
        }
        
        private double GradientDescent(List<Vector2> dataset, Func<double, double, double> func)
        {
            if(Logging) Console.WriteLine("Starting Gradient Descent........");
            
            /*
             * This function calculates how well optimized a parameter is in a function given a guess for the optimal value.
             *
             * This function takes in a dataset of x,y points and compares them to a mathematical function func and
             * calculates the loss of the func which tries to cross the x,y points as close as possible.
             */
            static double LossFunc(List<Vector2> data, Func<double, double, double> function,
                double initialParamGuess)
                => data.Sum(point => Math.Pow(point.Y - function(initialParamGuess, point.X), 2));

            double guess = initialGuess;
            double stepsTaken = 0;
            double step = 0;

            for (int i = 0; i < MaxSteps; i++)
            {
                if(Logging) Console.WriteLine($"Iteration: {stepsTaken} ");
                if(Logging) Console.WriteLine($"Step taken: {step}, guess: {guess}");
                
                // slope = (y2 - y1) / (x2 - x1) = deltaY / deltaX
                double dydxStep = guess + 0.001;
                double slope = 
                    (LossFunc(dataset, func, dydxStep) - LossFunc(dataset, func, guess)) / (dydxStep - guess);
                if(Logging) Console.WriteLine($"Slope of the Loss function: {slope}");
            
                step = -(slope * LearningRate);
                step = Math.Clamp(step, -MaxStepSize, MaxStepSize);

                if (Math.Abs(step) < MinStepSize)
                {
                    if(Logging) Console.WriteLine($"Minimum step size breached.  step: {step}, " +
                                                  $"minimum: {MinStepSize} minimum EXIT");
                    break;
                }

                stepsTaken++;

                if (stepsTaken >= MaxSteps)
                {
                    if(Logging) Console.WriteLine("Maximum steps taken breached. " +
                                                  $"Steps taken: {stepsTaken}, maximum steps: {MaxSteps} EXIT");
                    break;                   
                }
                
                guess += step;
            }
            
            if(Logging) Console.WriteLine($"Gradient descent finished. Guess is: {guess}");
            
            return guess;
        }

        private double CalculateMeanX()
        {
            return _dataset.Sum(point => point.X) / _dataset.Count;
        }
        
        private double CalculateMeanY()
        {
            return _dataset.Sum(point => point.Y) / _dataset.Count;
        }
    }
}