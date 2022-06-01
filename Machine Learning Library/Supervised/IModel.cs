namespace Machine_Learning_Library
{
    public interface IModel
    {
        void Fit(double independent, double dependent);
        
        double Predict(double dependent);
    }
}    