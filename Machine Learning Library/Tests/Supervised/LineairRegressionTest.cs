using NUnit.Framework;
using System;
using Machine_Learning_Library.Supervised;

namespace Machine_Learning_Library.tests
{
    [TestFixture]
    public class LineairRegressionTest
    {
        private readonly LineairRegression _negativeRelationModel;
        private readonly LineairRegression _positiveRelationModel;

        public LineairRegressionTest()
        {
            _negativeRelationModel = new LineairRegression();
            _negativeRelationModel.Fit(1, 4);
            _negativeRelationModel.Fit(2, 3);
            _negativeRelationModel.Fit(3, 3);
            _negativeRelationModel.Fit(4, 2);
            _negativeRelationModel.Fit(5, 1);
            _negativeRelationModel.Fit(6, 2);
            
            _positiveRelationModel = new LineairRegression();
            _positiveRelationModel.Fit(1, 2);
            _positiveRelationModel.Fit(2, 4);
            _positiveRelationModel.Fit(3, 5);
            _positiveRelationModel.Fit(4, 4);
            _positiveRelationModel.Fit(5, 5);
        }

        [Test]
        public void TestPositiveRelation()
        {
            _positiveRelationModel.InterceptStrategy = LineairRegression.InterceptStrategies.EquationSolving;

            Assert.AreEqual(4, _positiveRelationModel.Predict(3));
            
            Console.WriteLine(_positiveRelationModel.FunctionAsString());
        }

        [Test]
        public void TestNegativeRelation() {
            _negativeRelationModel.InterceptStrategy = LineairRegression.InterceptStrategies.EquationSolving;

            Assert.AreEqual(1.7714285714285718d, _negativeRelationModel.Predict(5));
            
            Console.WriteLine(_negativeRelationModel.FunctionAsString());
        }

        [Test]
        public void TestPositiveRelationWithGradientDescent()
        {
            _positiveRelationModel.InterceptStrategy = LineairRegression.InterceptStrategies.GradientDescent;
            _positiveRelationModel.GradDesc.Logging = true;
            
            _positiveRelationModel.GradDesc.LearningRate = 0.01;
            _positiveRelationModel.GradDesc.MaxSteps = 200;
            _positiveRelationModel.GradDesc.MinStepSize = 0.01;
            _positiveRelationModel.GradDesc.MaxStepSize = 0.25;
            _positiveRelationModel.GradDesc.InitialGuess = 7;
            
            Console.WriteLine(_positiveRelationModel.FunctionAsString());
        }
        
        [Test]
        public void TestNegativeRelationWithGradientDescent()
        {
            _negativeRelationModel.InterceptStrategy = LineairRegression.InterceptStrategies.GradientDescent;
            _negativeRelationModel.GradDesc.Logging = true;

            _negativeRelationModel.GradDesc.LearningRate = 0.01;
            _negativeRelationModel.GradDesc.MaxSteps = 50;
            _negativeRelationModel.GradDesc.MinStepSize = 0.01;
            _negativeRelationModel.GradDesc.MaxStepSize = 0.25;
            _negativeRelationModel.GradDesc.InitialGuess = 8;
            
            Console.WriteLine(_negativeRelationModel.FunctionAsString());
        }
    }
}