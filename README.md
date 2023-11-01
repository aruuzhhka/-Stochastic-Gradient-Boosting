# Қажетті кітапханаларды импорттау
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
# Деректер жиынтығын жүктеу
data = load_breast_cancer()
X = data.data
y = data.target
# Деректерді оқу және тест жиынтығына бөлу
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Деректерді стандарттау
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Градиентті күшейту моделін құру
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
# Модельді оқыту
model.fit(X_train, y_train)
# Сынақ деректерін болжау
y_pred = model.predict(X_test)
# Дәлдікті бағалау
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
# Жіктеу есебін шығару
print(classification_report(y_test, y_pred))
# Қате матрицасын шығару
confusion = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(confusion)
