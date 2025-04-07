import numpy as np

# Dataset

dados = np.array([
    [-1, -0.6508, 0.1097, 4.0009, -1],
    [-1, -1.4492, 0.8896, 4.4005, -1],
    [-1,  2.0850, 0.6876, 12.0710, -1],
    [-1,  0.2626, 1.1476, 7.7985, 1],
    [-1,  0.6418, 1.0234, 7.0427, 1],
    [-1,  0.2569, 0.6730, 8.3265, -1],
    [-1,  1.1155, 0.6043, 7.4446, 1],
    [-1,  0.0914, 0.3399, 7.0677, -1],
    [-1,  0.0121, 0.5256, 4.6316, 1],
    [-1, -0.0429, 0.4660, 5.4323, 1],
    [-1,  0.4340, 0.6870, 8.2287, -1],
    [-1,  0.2735, 1.0287, 7.1934, 1],
    [-1,  0.4839, 0.4851, 7.4850, -1],
    [-1,  0.4089, -0.1267, 5.5019, -1],
    [-1,  1.4391, 0.1614, 8.5843, -1],
    [-1, -0.9115, -0.1973, 2.1962, -1],
    [-1,  0.3654, 1.0475, 7.4858, 1],
    [-1,  0.2144, 0.7515, 7.1699, 1],
    [-1,  0.2013, 1.0014, 6.5489, 1],
    [-1,  0.6483, 0.2183, 5.8991, 1],
    [-1, -0.1147, 0.2242, 7.2435, -1],
    [-1, -0.7970, 0.8795, 3.8762, 1],
    [-1, -1.0625, 0.6366, 2.4707, 1],
    [-1,  0.5307, 0.1285, 5.6883, 1],
    [-1, -1.2200, 0.7777, 1.7252, 1],
    [-1,  0.3957, 0.1076, 5.6623, -1],
    [-1, -0.1013, 0.5989, 7.1812, -1],
    [-1,  2.4482, 0.9455, 11.2095, 1],
    [-1,  2.0149, 0.6192, 10.9263, -1],
    [-1,  0.2012, 0.2611, 5.4631, 1],
])

# Dataset: Separação de dados de Entrada X e saídas Y
X = dados[:, :-1]
Y = dados[:, -1].astype(int)

# Função de Ativação
def degrau_bipolar(u):
    return np.where(u > 0, 1, np.where(u < 0, -1, 0))

# Treino
def neuronio_d_treino(X, Yd, taxa_aprendizado=0.1, max_gerations=1000):
    n_amostras, n_entradas = X.shape
    pesos = np.random.uniform(-1, 1, size=n_entradas)
    print(f"\nPesos Aleatórios Gerados: {pesos} \n")
    
    for _ in range(max_gerations):
        erro_total = 0
        for i in range(n_amostras):
            entrada = X[i]
            esperado = Yd[i]
            u = np.dot(entrada, pesos)
            Yp = degrau_bipolar(u)
            erro = esperado - Yp
            pesos += taxa_aprendizado * erro * entrada
            erro_total += abs(erro)
        if erro_total == 0:
            break
    return pesos

# Teste
def neuronio_d_teste(X_test, Y_test, pesos):
    acertos = 0
    total = len(Y_test)
    for i in range(total):
        u = np.dot(X_test[i], pesos)
        Yp = degrau_bipolar(u)
        if Yp == Y_test[i]:
            acertos += 1
    # Formula Acurácia        
    acuracia = acertos / total
    return acuracia, acertos, total

# Embaralhar os dados
# np.random.seed(x) Teste de Valor Fixo para verificação, ignorar
indices = np.random.permutation(len(X))
X = X[indices]
Y = Y[indices]

# Separar 2/3 para Treino e 1/3 para Teste

tamanho_treino = int(len(X) * (2/3))
X_treino, Y_treino = X[:tamanho_treino], Y[:tamanho_treino]
X_teste, Y_teste = X[tamanho_treino:], Y[tamanho_treino:]

# Treino e Teste

pesos_finais = neuronio_d_treino(X_treino, Y_treino)
acuracia, acertos, total = neuronio_d_teste(X_teste, Y_teste, pesos_finais)

# Resultado

print("Resultado:")
print("---" * 30)
print(f"Pesos Finais: {pesos_finais}.")
print(f"Acurácia no teste: {acuracia:.2%} ({acertos}/{total} acertos). \n")