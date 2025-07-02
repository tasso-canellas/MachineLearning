# Regressão logística
# Dada uma reta como sei se um ponto está acima ou abaixo dela?
    # Nesse caso, para y > (-ax0-c)/b, ax0 + by + c > 0
    # Para y < (-ax0-c)/b, ax0 + by + c < 0

# Como funciona a cross-entropy?
    # Usam negativo do log da probabilidade pois a probabilidade está entre 0 e 1
    # y_pred é a probabilidade de x ser da classe 1
    
# Qual a derivida da cross-entropy?
    # (y_pred - y)*x

# Pq usar acuracia balanceada em caso de classificador binário?

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


def dados_logistica(n_samples:int = 100, n_features:int = 3, noise:int = 1) -> np.ndarray:
    X = np.random.rand(n_samples, n_features)*2-1#* 10
    theta = np.array([0]+ [1.0]*n_features)
    X_b = np.c_[np.ones((n_samples, 1)), X]
    logits = np.dot(X_b, theta) + noise*np.random.randn(n_samples)
    prob = 1 / (1 + np.exp(-logits))  
    y = np.where(prob > 0.5, 1, 0) 
    return X_b, y

def cross_entropy(y_pred, y):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15) 
    return -np.mean(y*np.log(y_pred) + (1 - y)*np.log(1 - y_pred))

def ativacao(x):
    return 1/(1+np.exp(-np.clip(x, -250, 250)))

def descida_estocastica_logistica(X_b, y, lr = 0.05, epochs = 100):
    n_features = X_b.shape[1]
    n_samples = X_b.shape[0]
    
    theta = np.random.rand(n_features) * 0.1 
    
    min_loss = 1e-6
    best_loss = float('inf')
    best_epoch = 0
    best_theta = theta.copy()
    best_acc = 0
    
    
    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        X_shuffled = X_b[indices]
        y_shuffled = y[indices]
        
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        epoch_loss = 0
        
        for i in range(n_samples):
            y_pred = np.dot(X_shuffled[i], theta)
            p = ativacao(y_pred)
            p_true = y_shuffled[i]
            
            if p > 0.5:
                if p_true == 1:
                    tp += 1
                else:
                    fp += 1
            else:
                if p_true == 0:
                    tn += 1
                else:
                    fn += 1
            
            p = np.clip(p, 1e-15, 1 - 1e-15)
            loss_individual = -(p_true * np.log(p) + (1 - p_true) * np.log(1 - p))
            epoch_loss += loss_individual
            
            grad = (p - p_true) * X_shuffled[i]
            theta = theta - lr * grad
        
        epoch_loss /= n_samples
        
        sensitivity = tp / (tp + fn + 1e-15)
        specificity = tn / (tn + fp + 1e-15)
        balanced_accuracy = (sensitivity + specificity) / 2
        
        if epoch % 10 == 0: 
            print(f'Epoch {epoch}, Balanced Acc: {balanced_accuracy:.4f}, Loss: {epoch_loss:.4f}')
        
        if epoch_loss < min_loss:
            return balanced_accuracy, epoch_loss, epoch, theta.copy()
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch
            best_acc = balanced_accuracy
            best_theta = theta.copy()
            
            
    return best_acc, best_loss, best_epoch, best_theta
           
if __name__ == '__main__':
    #X, y = dados_logistica(noise = 0.15)
    
    data = pd.read_csv("data.csv")
    data = data.iloc[:, :-1]
    data['diagnosis'] = data['diagnosis'].map({'B':0, 'M':1})
    
    X_train, X_test = train_test_split(data, test_size=0.3, stratify=data['diagnosis'], random_state=42)
    y_train = X_train['diagnosis'].values
    X_train = X_train.drop('diagnosis', axis=1).values
    y_test = X_test['diagnosis'].values
    X_test = X_test.drop('diagnosis', axis=1).values

    # Normalização dos dados
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    
    X_train = (X_train - mean) / (std + 1e-15)
    X_test= (X_test - mean) / (std + 1e-15)
    
    X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]
    
    print("Training data shape:", X_train_b.shape)
    print("Testing data shape:", X_test_b.shape)

    best_acc, best_custo, best_epoch, best_theta = descida_estocastica_logistica(X_train_b, y_train)
    print(f'Melhor acurácia: {best_acc}')
    print(f'Melhor custo: {best_custo}')
    print(f'Melhor epoca: {best_epoch}')
    
    # Testando o modelo nos dados de teste
    y_pred = ativacao(np.dot(X_test_b, best_theta))
    y_pred_classes = np.where(y_pred > 0.5, 1, 0)
    test_accuracy = np.mean(y_pred_classes == y_test)
    print(f'Test accuracy: {test_accuracy:.4f}')

            