import numpy as np
import matplotlib.pyplot as plt

def dados(n_samples:int = 100, n_features:int = 3, noise:int = 1) -> np.ndarray:
    #Feature scaling -> Normalização -> Base radial
    # min-max scaling
    # X = (X - X.min()) / (X.max() - X.min())
    # normalization scaling
    # X = (X - X.mean()) / (X.std()) para cada coluna independente
    # aplicar media e desvio padrão de treino na base de teste
    X = np.random.rand(n_samples, n_features) #* 10
    theta = np.array([5.0]+ [2.0]*n_features)
    X_b = np.c_[np.ones((n_samples, 1)), X]
    y = np.dot(X_b, theta) + noise*np.random.randn(n_samples)
    return X_b, y

def dados_logistica(n_samples:int = 100, n_features:int = 3, noise:int = 1) -> np.ndarray:
    X = np.random.rand(n_samples, n_features)*2-1#* 10
    theta = np.array([0]+ [1.0]*n_features)
    X_b = np.c_[np.ones((n_samples, 1)), X]
    logits = np.dot(X_b, theta) + noise*np.random.randn(n_samples)
    prob = 1 / (1 + np.exp(-logits))  
    y = np.where(prob > 0.5, 1, 0) 
    return X_b, y

def descida_estocastica(X_b, y, lr = 0.1, epochs = 1000):
    n_features = X_b.shape[1]
    n_samples = X_b.shape[0]
    
    theta = np.random.rand(n_features)
    
    best_custo = float('inf')
    custo_min = 1e-10
    best_epoch = 0
    best_theta = theta.copy()
    
    for epoch in range(epochs):
        epoch_custo = 0
        indices = np.random.permutation(n_samples)
        X_shuffled = X_b[indices]
        y_shuffled = y[indices]
        
        for i in range(n_samples):
            infe = np.dot(X_shuffled[i],theta)
            erro = infe-y_shuffled[i]
            
            gradientes = erro*X_shuffled[i]/n_samples
            
            theta = theta-lr*gradientes
            
            custo = (0.5)*(erro**2)
            epoch_custo += custo
            
        epoch_custo /= n_samples
            
        print(f'Epoch {epoch}, Cost: {epoch_custo}, Theta: {theta}')
        if epoch_custo < custo_min:
            return epoch_custo, epoch, theta.copy()
        if epoch_custo < best_custo:
            best_custo = epoch_custo
            best_epoch = epoch
            best_theta = theta.copy()
                 
    return best_custo, best_epoch, best_theta

def visualizar(X_b, y, theta):
    # Calcular as predições do modelo
    y_pred = np.dot(X_b, theta)
    
    
    # Plotar valores reais vs. preditos
    plt.figure(figsize=(8, 6))
    plt.scatter(y, y_pred, color='blue')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    plt.xlabel('Valores reais')
    plt.ylabel('Valores preditos')
    plt.title('Valores reais vs. Valores preditos')
    plt.grid(True)
    plt.show()
    
if __name__ == '__main__':
    X, y = dados(noise = 0)

    best_custo, best_epoch, best_theta = descida_estocastica(X, y)
    print(f'Melhor custo: {best_custo}')
    print(f'Melhor epoca: {best_epoch}')
    visualizar(X,y, best_theta)

