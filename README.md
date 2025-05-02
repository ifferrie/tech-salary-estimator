# tech-salary-estimator


# Tech Salary Estimator

This project utilizes a random forest model to estimate tech salaries based on the 2016 data from the Kaggle dataset [Know Your Worth: Tech Salaries in 2016](https://www.kaggle.com/datasets/thedevastator/know-your-worth-tech-salaries-in-2016).

## How to Run

1. Install the required packages:
   ```
   pip install sklearn pandas joblib streamlit
   ```

2. Build the model `.pkl` and evaluate its performance:
   ```
   python model.py
   ```

3. Run the web UI:
   ```
   streamlit run app.py
   ```

