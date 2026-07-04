import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import matplotlib.pyplot as plt

df = pd.read_csv("hyderabad_data_2.csv")
X = df.drop(["ST_B10","Unnamed: 9", "Unnamed: 10", "latitude", "longitude"], axis=1)
Y = df["ST_B10"]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)

rf = RandomForestRegressor(
    n_estimators=71,
    oob_score = True
)
rf.fit(X_train, Y_train)
y_pred = rf.predict(X_test)
r2 = r2_score(Y_test, y_pred)
rmse = root_mean_squared_error(Y_test, y_pred)
mae = mean_absolute_error(Y_test, y_pred)

score=[rf.oob_score_*100, r2*100]
tags = ["OOB Score(Out of Bag)", "R2"]
plt.bar(tags, score)
plt.xlabel("Evaluation Metric")
plt.ylabel("Percentage")

for i in range(len(score)):
  v=score[i]
  u=tags[i]
  v=int(v*100)
  v/=100
  plt.text(i,v+1,str(v),ha="center")
plt.tight_layout()
plt.show()

score=[rmse, mae]
tags = ["RMSE(Root Mean Squared Error)", "MAE(Mean Absolute Error)"]
plt.bar(tags, score)
plt.xlabel("Evaluation Metric")

for i in range(len(score)):
  v=score[i]
  u=tags[i]
  v=int(v*100)
  v/=100
  plt.text(i,v+0.02,str(v),ha="center")
plt.tight_layout()
plt.show()

x=["Temp", "Humidity", "Wind Speed", "NDVI", "NDBI", "NDWI"]
y=rf.feature_importances_
plt.bar(x,y*100)
plt.xlabel("Features")
plt.ylabel("Feature Importance(in Percent)")
for i,v in enumerate(y*100):
  v=int(v*100)
  v/=100
  plt.text(i,v+0.3,str(v),ha="center")
plt.tight_layout()
plt.show()

