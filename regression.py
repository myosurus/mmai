import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


file_path = 'ready_sales_dataset.csv'
data = pd.read_csv(file_path)


# Выбираем целевую переменную и числовые признаки для регрессии
target = 'TotalPrice'
numerical_features = ['UnitPrice', 'Discount', 'Quantity', 'ReturnStatus_Returned']

# Разделение на признаки и целевую переменную
X = data[numerical_features]
y = data[target]

# Разделение на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабирование данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_train_scaled[:5], y_train.head())


# Обучение модели линейной регрессии
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Предсказания на тестовом наборе
y_pred = lr_model.predict(X_test_scaled)

# Оценка результатов
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Коэффициенты модели линейной регрессии
coefficients = lr_model.coef_  # Коэффициенты при каждом признаке
intercept = lr_model.intercept_  # Свободный член

# Создаем таблицу для удобства анализа
regression_equation = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': coefficients
})

print("Свободный член (Intercept):", intercept)
print("\nКоэффициенты уравнения регрессии:")
print(regression_equation)


print("MAE: ", mae)
print("MSE: ", mse)
print("R2: ", r2)


# График: сравнение предсказанных и реальных значений
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color="blue", label="Предсказания")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label="Идеальная линия")
plt.xlabel("Реальные значения")
plt.ylabel("Предсказанные значения")
plt.title("Сравнение реальных и предсказанных значений (Linear Regression)")
plt.legend()
plt.grid(True)
plt.show()


input("\n\nWaiting to exit on button push...")
