# -*- coding: utf-8 -*-
# Ejercicio: Clasificador de SPAM usando Naive Bayes
# Autor: Juan Cruz
# --------------------------------------------------

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1️⃣ Datos de ejemplo (puedes reemplazarlos por un CSV o dataset real)
correos = [
    "oferta exclusiva compra ahora",
    "reunión de trabajo mañana a las 10",
    "gana dinero fácil desde casa",
    "informe semanal del proyecto",
    "urgente premio ganador",
    "recordatorio de pago pendiente",
    "viaje gratis a cancún",
    "actualización del sistema operativo",
    "promoción especial solo hoy",
    "tu cuenta ha sido bloqueada"
]

etiquetas = [
    "spam",
    "no_spam",
    "spam",
    "no_spam",
    "spam",
    "no_spam",
    "spam",
    "no_spam",
    "spam",
    "spam"
]

# 2️⃣ Convertimos texto a matriz numérica (Bolsa de Palabras)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(correos)

# 3️⃣ División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, etiquetas, test_size=0.3, random_state=42
)

# 4️⃣ Entrenamiento del modelo Naive Bayes Multinomial
modelo = MultinomialNB()
modelo.fit(X_train, y_train)

# 5️⃣ Predicción y evaluación
y_pred = modelo.predict(X_test)

print("🔹 Exactitud del modelo:", accuracy_score(y_test, y_pred))
print("\n🔹 Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))
print("\n🔹 Reporte de clasificación:")
print(classification_report(y_test, y_pred))

# 6️⃣ Probar con nuevos mensajes
nuevos_correos = [
    "compra este producto exclusivo",
    "te envío el informe del mes",
    "gana un viaje a europa gratis",
    "tu cuenta bancaria fue bloqueada"
]

X_nuevos = vectorizer.transform(nuevos_correos)
pred_nuevos = modelo.predict(X_nuevos)

print("\n📨 Predicciones en nuevos correos:")
for correo, pred in zip(nuevos_correos, pred_nuevos):
    print(f"- '{correo}' ➜ {pred}")
