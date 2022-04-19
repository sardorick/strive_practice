from train import rf_regressor, ab_regressor, gb_regressor, xg_regressor
from train import x_test, x_train, y_train, y_test


predict_1 = rf_regressor.predict(x_test)
predict_1_results = (predict_1 == y_test).sum() / (len(y_test))
print(f'The accuracy of the Random Forest regressor is {predict_1_results}')

predict_2 = ab_regressor.predict(x_test)
predict_2_results = (predict_2 == y_test).sum() / (len(y_test))
print(f'The accuracy of the Adaboost regressor is {predict_2_results}')

predict_3 = gb_regressor.predict(x_test)
predict_3_results = (predict_3 == y_test).sum() / (len(y_test))
print(f'The accuracy of the Gradient boost regressor is {predict_3_results}')

predict_4 = xg_regressor.predict(x_test)
predict_4_results = (predict_4 == y_test).sum() / (len(y_test))
print(f'The accuracy of the XG regressor is {predict_4_results}')