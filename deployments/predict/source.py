import modelbit, sys
from typing import *
from xgboost.sklearn import XGBClassifier

model = modelbit.load_value("data/model.pkl") # XGBClassifier(base_score=None, booster=None, callbacks=None, colsample_bylevel=None, colsample_bynode=None, colsample_bytree=None, device=None, early_stopping_rounds=None, enable_categorical=False, ev...

# main function
def predict(bloodpressure:float,
            cholesterol:float,
            smoker:float,
            age:float,
            sex:float,
            bmi:float,
            fruits:float,
            alcohol:float,
            heartattack:float,
            activity:float)->list:
            return model.predict([[bloodpressure, cholesterol, smoker, age, sex, bmi, fruits, alcohol, heartattack, activity]])

# to run locally via git & terminal, uncomment the following lines
# if __name__ == "__main__":
#   print(predict(*(modelbit.parseArg(v) for v in sys.argv[1:])))