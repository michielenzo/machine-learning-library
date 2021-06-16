using NUnit.Framework;
using System;

namespace Machine_Learning_Library.tests
{
    [TestFixture]
    public class LineairRegressionTest
    {
        private readonly LineairRegression _lineairRegression;

        public LineairRegressionTest()
        {
            _lineairRegression = new LineairRegression();
        }

        [Test]
        public void TestPositiveRelation()
        {
            _lineairRegression.Fit(1, 2);
            _lineairRegression.Fit(2, 4);
            _lineairRegression.Fit(3, 5);
            _lineairRegression.Fit(4, 4);
            _lineairRegression.Fit(5, 5);

            Assert.AreEqual(4, _lineairRegression.Predict(3));

            _lineairRegression.Unfit();
        }

        [Test]
        public void TestNegativeRelation() {
            _lineairRegression.Fit(1, 4);
            _lineairRegression.Fit(2, 3);
            _lineairRegression.Fit(3, 3);
            _lineairRegression.Fit(4, 2);
            _lineairRegression.Fit(5, 1);
            _lineairRegression.Fit(6, 2);

            Assert.AreEqual(1.7714283466339111f, _lineairRegression.Predict(5));

            _lineairRegression.Unfit();
        }
    }
}