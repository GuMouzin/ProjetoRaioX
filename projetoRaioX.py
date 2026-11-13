import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Caminhos locais para o dataset (pastas na raiz)
caminhoTreino = "treino"
caminhoTeste = "teste"
caminhoProducao = "producao"

# Geradores de imagens com aumento de dados para treino
gerador_treino = ImageDataGenerator(
    rescale=1./255, # Redimensionamento dos pixels para o intervalo [0,1]
    shear_range=0.2, # Transformação de cisalhamento
    zoom_range=0.2, # Intervalo de zoom aleatório
    horizontal_flip=True # Rotação horizontal aleatória
)

# Gerador apenas com redimensionamento para teste
gerador_teste = ImageDataGenerator(
    rescale=1./255
)

dados_treino = gerador_treino.flow_from_directory(
    caminhoTreino,
    target_size=(224, 224), # Tamanho das imagens de entrada
    batch_size=32, # Tamanho do lote
    class_mode='binary' # Modo de classificação binária
)

dados_teste = gerador_teste.flow_from_directory(
    caminhoTeste,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Construção da Rede Neural Convolucional (CNN)
modelo = tf.keras.models.Sequential([
    Input(shape=(224, 224, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Exibir arquitetura da rede
print(modelo.summary())

# Compilar o modelo
modelo.compile(optimizer='adam',
               loss='binary_crossentropy',
               metrics=['accuracy'])

# Treinar o modelo
historicoAprendizado = modelo.fit(dados_treino, epochs=10, validation_data=dados_teste)

# Plotar gráficos de aprendizado
pd.DataFrame(historicoAprendizado.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0, 5)
plt.show()

# Avaliar o modelo com dados de teste
precisao = modelo.evaluate(dados_teste)[1]
print("A precisão do modelo é:", precisao)

# Testar o modelo com uma imagem de exemplo na pasta producao
caminhoImagem = "producao/V-01.jpeg"
try:
    imagemTeste = load_img(caminhoImagem, target_size=(224, 224))
    imagemArray = img_to_array(imagemTeste)
    imagemArray = np.expand_dims(imagemArray, axis=0)
    imagemArray = imagemArray / 255.0  # Normalização

    # Fazer previsão com o modelo
    previsao = modelo.predict(imagemArray)

    # Imprimir a previsão
    if previsao[0][0] > 0.5:
        print("A imagem é classificada como 'doente'.")
    else:
        print("A imagem é classificada como 'normal'.")
except Exception as e:
    print(f"Não foi possível carregar a imagem de teste ({caminhoImagem}): {e}")