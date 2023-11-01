# -Stochastic-Gradient-Boosting
# Қажетті кітапханаларды импорттау
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
# Мысал үшін кездейсоқ деректерді жасаңыз
np.random.seed(0)
num_customers = 1000
gender = np.random.choice(['Male', 'Female'], num_customers)
age = np.random.randint(18, 65, num_customers)
style = np.random.choice(['Casual', 'Formal', 'Sportswear'], num_customers)
budget = np.random.choice(['Low', 'Medium', 'High'], num_customers)
clothing_category = np.random.choice(['Shirts', 'Jeans', 'Dresses', 'Suits'], num_customers)
# Мақсатты айнымалыны жасаңыз - киімдегі артықшылықтар (бұл мысалда кездейсоқ мәндер)
clothing_preference = np.random.choice(['Yes', 'No'], num_customers)
# Сатып алушылар туралы белгілер жасау
data = pd.DataFrame({'gender': gender, 'age': age, 'style': style, 'budget': budget, 'clothing_category': clothing_category})
#Категориялық белгілерді one-hot encoding көмегімен сандық форматқа түрлендіреміз
data = pd.get_dummies(data, columns=['gender', 'style', 'budget', 'clothing_category'])
#деректерді оқу және тест жиынтықтарына бөлеміз
X_train, X_test, y_train, y_test = train_test_split(data, clothing_preference, test_size=0.2, random_state=42)
# Жіктеу үшін Stochastic Gradient Boosting моделін жасау
classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
#модельді оқыту деректер жиынтығында оқытамыз
classifier.fit(X_train, y_train)
#тест деректері бойынша сатып алушылар үшін киімге артықшылық береміз
y_pred = classifier.predict(X_test)
# Модельдің сапасын дәлдікпен бағалау
accuracy = accuracy_score(y_test, y_pred)
print("Модель далдиги:", accuracy)
