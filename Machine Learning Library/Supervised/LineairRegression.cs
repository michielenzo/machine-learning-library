using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;

namespace Machine_Learning_Library
{
    public class LineairRegression: IModel
    {

        private readonly List<Vector2> _dataset;

        public LineairRegression()
        {
            _dataset = new List<Vector2>();
        }

        public void Fit(float independent, float dependent)
        {
            _dataset.Add(new Vector2(independent, dependent));
        }

        public double Predict(float dependent)
        {
            return CalculateIntercept() + CalculateSlope() * dependent;
        }

        public void Unfit() {
            _dataset.Clear();
        }

        private float CalculateSlope()
        {
            float meanX = CalculateMeanX();
            float meanY = CalculateMeanY();

            float sumXyDistanceToMean = _dataset.Sum(point => (point.X - meanX) * (point.Y - meanY));
            float sumXDistanceToMean = _dataset.Sum(point => (float) Math.Pow(point.X - meanX, 2));

            return sumXyDistanceToMean / sumXDistanceToMean;
        }

        private float CalculateIntercept()
        {
            return -(CalculateSlope() * CalculateMeanX() - CalculateMeanY());
        }

        private float CalculateMeanX()
        {
            return _dataset.Sum(point => point.X) / _dataset.Count;
        }
        
        private float CalculateMeanY()
        {
            return _dataset.Sum(point => point.Y) / _dataset.Count;
        }
    }
}