# Machine Learning Capstone

A comprehensive machine learning project focused on predicting housing prices using linear regression. This capstone project demonstrates data cleaning, exploratory data analysis, model training, and interactive price prediction.

## 📋 Project Overview

This project analyzes a housing dataset to predict property prices based on various features such as area, number of bedrooms/bathrooms, amenities, and location characteristics. The implementation includes data visualization, correlation analysis, and a trained linear regression model with an interactive prediction interface.

## 🎯 Features

- **Data Cleaning & Preprocessing**: Conversion of categorical variables to numerical format using ordinal encoding
- **Exploratory Data Analysis**: Distribution plots, correlation heatmaps, and scatter plot visualizations
- **Linear Regression Model**: Trained model with coefficient analysis and feature importance insights
- **Model Evaluation**: Performance metrics including MAE and MSE
- **Interactive Price Predictor**: Widget-based interface for real-time house price predictions

## 📊 Dataset

The project uses a housing dataset (`housingData.csv`) with the following features:

- **price**: Target variable (house price)
- **area**: Square footage of the property
- **bedrooms**: Number of bedrooms
- **bathrooms**: Number of bathrooms
- **stories**: Number of floors
- **mainroad**: Access to main road (yes/no)
- **guestroom**: Presence of guest room (yes/no)
- **basement**: Presence of basement (yes/no)
- **hotwaterheating**: Hot water heating system (yes/no)
- **airconditioning**: Air conditioning system (yes/no)
- **parking**: Number of parking spaces
- **prefarea**: Located in preferred area (yes/no)
- **furnishingstatus**: Furnished, semi-furnished, or unfurnished

## 🔧 Technologies Used

- **Python 3.8+**
- **pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib**: Data visualization
- **Seaborn**: Statistical data visualization
- **scikit-learn**: Machine learning library
  - `LinearRegression`: Model implementation
  - `train_test_split`: Data splitting
  - `OrdinalEncoder`: Categorical encoding
  - Metrics: MAE, MSE
- **ipywidgets**: Interactive widgets for Jupyter notebooks
- **Google Colab**: Cloud-based notebook environment

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn ipywidgets
```

### Running the Notebook

1. Open the notebook in Google Colab:
   - Click the "Open in Colab" badge at the top of `capFinal.ipynb`
   
2. Mount Google Drive (optional):
   - Follow the authentication prompts to access datasets from Drive

3. Download the dataset:
   - The notebook includes code to download `housingData.csv` automatically

4. Run all cells sequentially to:
   - Load and clean the data
   - Generate visualizations
   - Train the linear regression model
   - Use the interactive price predictor

## 📈 Model Performance

The linear regression model was trained on a 60/40 train-test split. Key findings:

- **Strong Positive Correlations**: Area, bathrooms, and air conditioning show high correlation with price
- **Feature Coefficients**: Each bathroom adds approximately $1.05 million to the predicted price
- **Model Evaluation**: Performance measured using Mean Absolute Error (MAE) and Mean Squared Error (MSE)

## 🎨 Visualizations

The project includes several key visualizations:

1. **Price Distribution**: Histogram showing most houses priced between $2M-$6M
2. **Correlation Heatmap**: Shows relationships between all features
3. **Scatter Plots**: Visualizes the relationship between area and price
4. **Prediction Plot**: Compares actual vs. predicted prices with line of best fit

## 🔮 Interactive Predictor

The notebook features an interactive widget-based interface where you can:

- Adjust house features using sliders (area, bedrooms, bathrooms, etc.)
- Toggle amenities (parking, AC, basement, etc.)
- Click "Predict Price" to get instant price estimates based on the trained model

## 📝 Key Insights

- Average house price in the dataset: Calculated and displayed
- Price range: Most properties fall between $2M-$6M
- Most influential features: Area, bathrooms, and air conditioning
- Model interpretation: Coefficients provide actionable insights into feature impact

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Travis Hipolito**

- GitHub: [@Travis-Code](https://github.com/Travis-Code)

## 🎓 Acknowledgments

This project was completed as a capstone project for college coursework, demonstrating practical application of machine learning concepts in real estate price prediction.

---

*This project uses Google Colab for cloud-based execution and includes integration with Google Drive for data storage and retrieval.*
