import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error,r2_score

df =pd.read_csv("HousingData.csv")

print(df.head())
print(df.info())
print(df.describe())

plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), annot = True, cmap = 'coolwarm')
plt.title("feature correlation with MEDV")
plt.show()

X = df[["RM"]]
y = df["MEDV"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)

print("\n[linear regression]")
print("slope:", lr_model.coef_[0])
print("y-intercept", lr_model.intercept_)

plt.scatter(X_test, y_test, color = 'blue', label = 'true')
plt.plot(X_test,y_pred,color = 'red', label = 'esteminate')
plt.xlabel("RM")
plt.ylabel("home priice")
plt.title("linear regression estamiate")
plt.legend()
plt.show()

mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test, y_pred)
print("MSE", mse)
print("R2",r2)

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("real price")
plt.ylabel("estaminate price")
plt.title("real vs estaminated")
plt.grid(True)
plt.show()

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)

print("\n[Ridge Regression]")
print("R²:", r2_score(y_test, ridge_pred))

lasso = Lasso(alpha=0.1)
lasso.fit(X_train,y_train)
lasso_pred = lasso.predict(X_test)
print("\n [lasso regression]")
print("r2" , r2_score(y_test, lasso_pred))