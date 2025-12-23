# focused try for YXY with larger models
from pathlib import Path
import numpy as np, pandas as pd, joblib, json
BASE = Path(__file__).resolve().parents[1]
CLEAN_DIR = BASE / 'data' / 'cleaned_data'
MO_DIR = BASE / 'data' / 'model_objects'
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import importlib.util
spec = importlib.util.spec_from_file_location('adv', BASE / 'scripts' / 'advanced_train.py')
adv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adv)

def metrics(y_true,y_pred):
    mse=mean_squared_error(y_true,y_pred); return mse, float(np.sqrt(mse)), float(r2_score(y_true,y_pred))

path = CLEAN_DIR / 'YXY.csv'
df = pd.read_csv(path, encoding='utf-8-sig')
X, y = adv.build_features(df, use_poly=True)
mask = ~np.isnan(X.to_numpy()).any(axis=1)
X = X.loc[mask]; y = y[mask]
# remove outliers
ym = y.mean(); ys=y.std(); keep=(y>=ym-3*ys)&(y<=ym+3*ys)
X=X.loc[keep]; y=y[keep]
# subsample larger fraction
from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2, random_state=42)
# models
hgb = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_leaf_nodes=63, random_state=42)
rf = RandomForestRegressor(n_estimators=500, max_depth=30, n_jobs=-1, random_state=42)
ridge = Ridge(alpha=1.0)
estimators=[('hgb',hgb),('rf',rf),('ridge',Pipeline([('scaler',StandardScaler()),('ridge',ridge)]))]
from sklearn.ensemble import StackingRegressor
stack = StackingRegressor(estimators=estimators, final_estimator=LinearRegression(), passthrough=True, n_jobs=-1)
# try transforms
best=None
for t in ('none','log1p'):
    if t=='none': yt=y_train.copy(); yv=y_val.copy()
    else: yt=np.log1p(y_train); yv=np.log1p(y_val)
    try:
        stack.fit(X_train.values, yt)
        predt = stack.predict(X_val.values)
        if t=='log1p': pred = np.expm1(predt); yv_orig = np.expm1(yv)
        else: pred=predt; yv_orig=yv
        mse,rmse,r2=metrics(yv_orig,pred)
        print('transform', t, 'R2', r2, 'rmse', rmse)
        if best is None or r2>best[1]: best=(t,r2,rmse,stack)
    except Exception as e:
        print('failed', e)

print('best', best)
if best and best[1]>=0.5:
    joblib.dump({'model':best[3], 'features': list(X.columns)}, MO_DIR / 'focused_YXY_model.joblib')
    print('saved model')
else:
    print('did not reach target')

