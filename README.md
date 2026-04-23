# 💼 Employee Salary Predictor

> A beginner-friendly, end-to-end Machine Learning project that predicts an employee's **Annual Salary** based on professional and educational attributes — built with Python, Scikit-learn, and Streamlit.

---

## 📌 Description

This project walks through a **complete supervised ML pipeline**:

- Synthetic but realistic dataset of 500 employee records
- Full data cleaning and exploratory data analysis (EDA)
- Linear Regression model trained with Scikit-learn
- Model evaluation using MAE, MSE, RMSE, and R² Score
- Interactive Streamlit web app for real-time salary prediction

The trained model achieves an **R² score of 98.70%** on the test set.

---

## 🗂️ Project Structure

```
salary-predictor/
│
├── app.py              # Streamlit web application (UI)
├── model.py            # ML pipeline: data → train → evaluate → save
├── dataset.csv         # Synthetic employee dataset (500 rows)
├── model.pkl           # Saved Linear Regression model (pickle)
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

## 🧰 Tech Stack

| Tool            | Purpose                          |
|-----------------|----------------------------------|
| Python 3.8+     | Core programming language        |
| Pandas          | Data loading and manipulation    |
| NumPy           | Numerical operations             |
| Scikit-learn    | ML model (Linear Regression)     |
| Streamlit       | Interactive web UI               |
| Pickle          | Model serialisation              |

---

## 📊 Dataset

| Column              | Type    | Description                             |
|---------------------|---------|-----------------------------------------|
| `Years_Experience`  | int     | Total years of work experience (0–30)   |
| `Age`               | int     | Employee age in years (22–57)           |
| `Education_Level`   | int     | 0 = Bachelor, 1 = Master, 2 = PhD       |
| `Hours_Per_Week`    | int     | Average weekly hours worked (35–60)     |
| `Num_Projects`      | int     | Number of projects handled (1–15)       |
| `Annual_Salary_INR` | float   | **Target** — Annual salary in INR (₹)   |

---

## 🧠 Model Performance

| Metric | Value        |
|--------|-------------|
| MAE    | ₹22,351     |
| MSE    | ₹92,36,63,510 |
| RMSE   | ₹30,392     |
| R²     | 0.9870 (98.70%) |

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/salary-predictor.git
cd salary-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model (generates `model.pkl`)
```bash
python model.py
```

### 4. Launch the Streamlit app
```bash
streamlit run app.py
```

The app will open at **http://localhost:8501** in your browser.

---

## 🖥️ Example Output

**Input:**
- Years of Experience: 8
- Age: 31
- Education Level: Master's (1)
- Hours Per Week: 45
- Number of Projects: 7

**Predicted Annual Salary: ₹8,65,692**

---

## 📸 App UI

> The app is simple and clean:
> - Fill in 5 inputs (Experience, Age, Education, Hours, Projects)
> - Click **Predict Salary**
> - Instantly see **₹ Annual Salary** + Monthly & Weekly breakdown
> - Expandable sections for Model Metrics and Dataset preview

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

[MIT](https://choosealicense.com/licenses/mit/)

---

*Built with ❤️ using Python · Scikit-learn · Streamlit*
