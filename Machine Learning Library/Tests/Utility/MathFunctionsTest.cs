using System;
using System.Collections.Generic;
using System.Numerics;
using Machine_Learning_Library.DeepLearning.NeuralNetwork;
using NUnit.Framework;

namespace Machine_Learning_Library.Tests.Utility
{
    [TestFixture]
    public class MathFunctionsTest
    {

        [Test]
        public void TestGetSlope()
        {
            double YEqualsXSquared(double x) => Math.Pow(x, 2);

            double slope = MathFunctions.GetSlope(YEqualsXSquared, 1.5);

            Assert.AreEqual(slope, 3.0010000000001398d);
        }
    }
}