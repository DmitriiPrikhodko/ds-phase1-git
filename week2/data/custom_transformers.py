from sklearn.base import BaseEstimator, TransformerMixin

# Создание своего Imputer'а
class AgeImputer(BaseEstimator, TransformerMixin): 
    def fit(self, X, y=None): # Запутстится в момент, когда будет вызван у всего pipeline метод .fit()
        self.params_ = X.groupby('Pclass')['Age'].mean().to_dict() # Сохраним в атрибуте экземпляра средние значения возраста по классам
        return self # метод .fit всегда должен возвращать сам экземпляр
    def transform(self, X, y=None): # Запустится тогда, когда нужно будет отдать результат на следующий шаг
        X_copy = X.copy()
        X_copy['Age'] = X_copy['Age'].fillna(X['Pclass'].map(self.params_))
        return X_copy[['Pclass', 'Age']]
    

# Создание своего преобразователя
class FamiliyTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_copy = X.copy()
        X_copy['number_of_family_members'] = X_copy['SibSp'] + X_copy['Parch']
        X_copy.drop(['SibSp', 'Parch'], axis=1, inplace=True)
        return X_copy