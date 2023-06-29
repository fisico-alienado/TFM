import numpy as np
import matplotlib.pyplot as plt
import h5py


def cargar_dataset():
    train_dataset = h5py.File('datos/train_gato_vs_otros.h5', "r")
    # características del conjunto de entrenamiento
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    # etiquetas del conjunto de entrenamiento
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('datos/test_gato_vs_otros.h5', "r")
    # características del conjunto de prueba
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) 
    # etiquetas del conjunto de pruebas
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])
    
    # listado de las clases
    clases = np.array(test_dataset["list_classes"][:]) 
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, clases

def inicializar_parametros(n_x, n_h, n_y):
    """
    Argumentos:
    n_x -- tamaño de la capa de entrada
    n_h -- tamaño de la capa oculta
    n_y -- tamaño de la capa de salida
    
    Devuelve:
    parametros -- diccionario python que contiene los parámetros:
                    W1 -- matriz de pesos de la forma (n_h, n_x)
                    b1 -- vector de sesgos de la forma (n_h, 1)
                    W2 -- matriz de pesos de la forma (n_y, n_h)
                    b2 -- vector de sesgos de la forma (n_y, 1)                   
    """   
    np.random.seed(1)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros(shape=(n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros(shape=(n_y, 1))    

    # Asegurar que las dimensiones son las correctas
    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))
    
    parametros = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parametros

def sigmoide(Z):
    """
    Calcula el sigmoide de Z
    
    Argumentos:
    Z -- array de NumPy de cualquier dimension (salida lineal de la capa)
    
    Devuelve:
    A -- resultado de sigmoide(Z), de las mismas dimensiones que Z.
    cache -- devuelve el argumento de entrada, Z, que se utilizara en la retropropagacion.
    """
    
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    
    return A, cache

def relu(Z):
    """
    Implementa la funcion ReLU.

    Argumentos:
    Z -- array de NumPy de cualquier dimension (salida lineal de la capa)

    Returns:
    A -- resultado de aplicar relu(Z), de las mismas dimensiones que Z. Es el parametro post-activacion.
    cache -- devuelve el argumento de entrada, Z, que se utilizara en la retropropagacion.
    """
    
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache

def forward_lineal(A, W, b):
    """
    Implementa la parte lineal de la propagación hacia delante de una capa.

    Argumentos:
    A -- activaciones de la capa anterior (o datos de entrada): (tamaño de la capa anterior, numero de ejemplos).
    W -- matriz de pesos: matriz numpy de dimensiones (tamaño de la capa actual, tamaño de la capa anterior).
    b -- vector de sesgos: matriz numpy de dimensiones (tamaño de la capa actual, 1).

    Devuelve:
    Z -- la entrada de la función de activación, tambien llamado parametro de pre-activacion .
    cache -- una tupla python que contiene "A", "W" y "b" ; almacenada para calcular eficientemente la retropropagación.
    """
    
    Z = np.dot(W, A) + b
    # Aseguramos que las dimensiones son correctas
    assert(Z.shape == (W.shape[0], A.shape[1]))
    
    cache = (A, W, b)
    
    return Z, cache

def forward_activacion_lineal(A_prev, W, b, activacion):
    """
    Implementar la propagación hacia delante para la capa de la forma LINEAL->ACTIVACIÓN

    Argumentos:
    A_prev -- activaciones de la capa anterior (o datos de entrada): (tamaño de la capa anterior, numero de ejemplos)
    W -- matriz de pesos: matriz numpy de forma (tamaño de la capa actual, tamaño de la capa anterior)
    b -- vector de sesgo, matriz numpy de forma (tamaño de la capa actual, 1)
    activacion -- la funcion de activacion a utilizar en esta capa, almacenada como cadena de texto: "sigmoide" o "relu".

    Devuelve:
    A -- la salida de la funcion de activacion, tambien llamada valor post-activación 
    cache -- una tupla python que contiene "cache_lineal" y "activacion_cache";
             almacenada para calcular la retropropagacion eficientemente
    """
    
    if activacion == "sigmoide":
        Z, cache_lineal = forward_lineal(A_prev, W, b)
        A, activacion_cache = sigmoide(Z)
    elif activacion == "relu":
        Z, cache_lineal = forward_lineal(A_prev, W, b)
        A, activacion_cache = relu(Z)        
    else:
        print("¡Error! Solo se admiten relu o sigmoide como parametros en \"activacion\"")
    
    # Aseguramos dimensiones
    assert(A.shape == (W.shape[0], A_prev.shape[1]))

    cache = (cache_lineal, activacion_cache)

    return A, cache

def calcular_coste(AL, Y):
    """
    Calcula el coste de la entropía cruzada dado en la ecuación (3).

    Argumentos:
    AL -- La salida de la activación de la capa L, de dimensiones (1, numero de casos de entrenamiento).
          Corresponde con el vector de probabilidades asociado a las etiquetas verdaderas, de las mismas dimensiones.
    Y -- Vector de etiquetas verdaderas de dimensiones (1, número de casos de entrenamiento).

    Devuelve:
    coste -- coste de la entropia cruzada dada la ecuacion (3).
    """
    
    # Numero de casos de entrenamiento
    m = Y.shape[1]

    # Calculo de la perdida de entropia cruzada
    # Forma 1
    # perdida = np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL))
    # coste = (-1 / m) * np.sum(perdida)

    # Forma 2
    coste = (1. / m) * (- np.dot(Y, np.log(AL).T) - np.dot(1 - Y, np.log(1 - AL).T))
    coste = np.squeeze(coste) # Por ejemplo, convierte [[17]] en 17
    
    assert(coste.shape == ())

    return coste

def relu_backward(dA, cache):
    """
    Implementa la propagación hacia atrás para una unica unidad ReLU.

    Argumentos:
    dA -- gradiente post-activación, de cualquier dimension.
    cache -- Es la variable 'Z' que fue almacenada en 'cache' para hacer un calculo de la retropropagacion eficiente.

    Devuelve:
    dZ -- Gradiente del coste con respecto a Z.
    """
    
    Z = cache
    # Hacemos una copia profunda de dA con 'copy=True', argumento que crea una copia completa de un objeto y todos sus elementos.
    # En el caso de una matriz, esto significa que se crea una nueva matriz con los mismos valores que la matriz original, 
    # pero en una ubicación de memoria diferente. Esto significa que si cambias un valor en la matriz original, 
    # no afectará a la copia profunda.
    dZ = np.array(dA, copy=True)
    
    # Cuando Z = 0, podemos decidir si queremos que equivalga a 0 o a 1. En este caso, a 0.
    dZ[Z <= 0] = 0
    
    # Clave asegurar que las dimensiones son correctas
    assert(dZ.shape == Z.shape)
    
    return dZ

def sigmoide_backward(dA, cache):
    """
    Implementa la propagación hacia atrás para una unica unidad ReLU.

    Argumentos:
    dA -- gradiente post-activación, de cualquier dimension.
    cache -- Es la variable 'Z' que fue almacenada en 'cache' para hacer un calculo de la retropropagacion eficiente.

    Devuelve:
    dZ -- Gradiente del coste con respecto a Z.
    """
    
    Z = cache
    # Sigmoide va a ser la ultima capa. dZ es la derivada del coste con respecto a Z de la ultima capa, es decir, dJ/dZ y, 
    # segun la regla de la cadena, es equivalente a:
    #       dZ = dJ/dZ = dJ/dA * dA/dZ, donde dJ/dA = dA, y donde dA/dZ = derivada de la funcion 'sigmoide(Z)', 
    # y la derivada del sigmoide es: s'(z) = s(z)*(1-s(z))
    # Funcion sigmoide
    s = 1 / (1 + np.exp(-Z))
    # dJ/dZ
    dZ = dA * s * (1 - s)
    
    # Clave asegurar que las dimensiones son correctas
    assert(dZ.shape == Z.shape)
    
    return dZ

def backward_lineal(dZ, cache):
    """
    Implementa la parte lineal de la retropropagación para una sola capa (capa l)

    Argumentos:
    dZ -- Gradiente del coste con respecto a la salida lineal (de la capa l actual).
    cache -- tupla de valores (A_prev, W, b) procedentes de forward_lineal() en la capa actual

    Devuelve:
    dA_prev -- Gradiente del coste con respecto a la activación (de la capa anterior l-1); mismas dimensiones que A_prev
    dW -- Gradiente del coste con respecto a W (capa actual l), mismas dimensiones que W
    db -- Gradiente del coste con respecto a b (capa actual l), mismas dimensiones que b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T)/m
    db = np.sum(dZ, axis=1, keepdims=True)/m
    dA_prev = np.dot(W.T, dZ)    
    
    # Aseguramos que las dimensiones sean correctas
    assert(dA_prev.shape == A_prev.shape)
    assert(dW.shape == W.shape)
    assert(db.shape == b.shape)    
    
    return dA_prev, dW, db

def backward_activacion_lineal(dA, cache, activacion):
    """
    Implementar la retropropagacion para la capa de la forma LINEAL->ACTIVACION
    
    Argumentos:
    dA -- gradiente post-activación para la capa actual l 
    cache -- tupla de valores (cache_lineal, activacion_cache) almacenados para calcular la propagación hacia atrás eficientemente
    activation -- la funcion de activacion a utilizar en esta capa, almacenada como cadena de texto: "sigmoide" o "relu".
    
    Devuelve
    dA_prev -- Gradiente del coste con respecto a la activación (de la capa anterior l-1); mismas dimensiones que A_prev
    dW -- Gradiente del coste con respecto a W (capa actual l), mismas dimensiones que W
    db -- Gradiente del coste con respecto a b (capa actual l), mismas dimensiones que b
    """
    cache_lineal, activacion_cache = cache
    
    if activacion == "relu":
        dZ = relu_backward(dA, activacion_cache)
        dA_prev, dW, db = backward_lineal(dZ, cache_lineal)
    elif activacion == "sigmoide":
        dZ = sigmoide_backward(dA, activacion_cache)
        dA_prev, dW, db = backward_lineal(dZ, cache_lineal)
    else:
        print("¡Error! Solo se admiten relu o sigmoide como parametros en \"activacion\"")
    
    return dA_prev, dW, db

def actualizar_parametros(parametros_entrada, gradientes, tasa_aprendizaje):
    """
    Actualiza los parámetros utilizando la regla de actualización por descenso de gradiente.
    
    Argumentos:
    parametros_entrada -- diccionario python que contiene los parametros 
    gradientes -- diccionario python que contiene los gradientes
    tasa_aprendizaje -- hiperparametro que representa la tasa de aprendizaje utilizada en la regla de actualizacion
    
    Devuelve:
    parametros -- diccionario python que contiene los parametros actualizados 
                  parametros["W" + str(l)] = ... 
                  parametros["b" + str(l)] = ...
    """
    # Recupera una copia de cada parámetro del diccionario "parametros_entrada".
    parametros = parametros_entrada.copy()
    # numero de capas de la red neuronal (para cada l-esima capa hay un parametro W y otro b)
    L = len(parametros) // 2 

    # Aplicar regla de actualizacion a cada parametro
    for l in range(L):
        parametros["W" + str(l + 1)] = parametros["W" + str(l + 1)] - tasa_aprendizaje * gradientes["dW" + str(l + 1)]
        parametros["b" + str(l + 1)] = parametros["b" + str(l + 1)] - tasa_aprendizaje * gradientes["db" + str(l + 1)]        

    return parametros

def inicializacion_profunda(dims_capas):
    """
    Argumentos:
    dims_capas -- array python (lista) que contiene las dimensiones de cada capa de nuestra red.
    
    Devuelve:
    parametros -- diccionario python que contiene los parámetros "W1", "b1", ..., "WL", "bL":
                    Wl -- matriz de pesos de dimensiones (dims_capas[l], dims_capas[l-1])
                    bl -- vector de sesgo de dimensiones (dims_capas[l], 1)
    """
    
    np.random.seed(1)
    parametros = {}
    # Numero de capas en la red neuronal
    L = len(dims_capas)

    for l in range(1, L):
        # Multiplicar por 0.01 ya no es suficiente con un modelo mas grande
        parametros['W' + str(l)] = np.random.randn(dims_capas[l], dims_capas[l - 1]) / np.sqrt(dims_capas[l-1]) #*0.01
        parametros['b' + str(l)] = np.zeros((dims_capas[l], 1))        
        
        # Asegurar que las dimensiones son correctas
        assert(parametros['W' + str(l)].shape == (dims_capas[l], dims_capas[l - 1]))
        assert(parametros['b' + str(l)].shape == (dims_capas[l], 1))
        
    return parametros

def propagacion_L_capas(X, parametros):
    """
    Implementa la propagación hacia delante para el cálculo [LINEAL->RELU]*(L-1)->LINEAL->SIGMOIDE
    
    Argumentos:
    X -- datos, array numpy de dimensiones (caracteristicas por ejemplo, número de ejemplos)
    parametros -- salida de inicializacion_profunda()
    
    Devuelve:
    AL -- valor de activación de la capa de salida
    caches -- lista de caches que contienen cada cache de forward_activacion_lineal() (hay L, indexadas de 0 a L-1)
    """

    caches = []
    # La activacion inicial, A[0], es la capa de entrada
    A = X
    # numero de capas de la red neuronal (para cada l-esima capa hay un parametro W y otro b)
    L = len(parametros) // 2
    
    # Implementar [LINEAL -> RELU]*(L-1). Añadimos "cache" a la lista de "caches".
    # El bucle comienza en 1 porque la capa 0 es la de entrada
    for l in range(1, L):
        A_prev = A 
        A, cache = forward_activacion_lineal(A_prev, 
                                             parametros['W' + str(l)], 
                                             parametros['b' + str(l)], 
                                             activacion="relu")
        caches.append(cache)        
    
    # Implementar LINEAL -> SIGMOIDE. Añadimos "cache" a la lista de "caches".
    AL, cache = forward_activacion_lineal(A, 
                                          parametros['W' + str(L)], 
                                          parametros['b' + str(L)], 
                                          activacion="sigmoide")
    caches.append(cache)
    
    # Nos aseguramos que las dimensiones sean correctas (n_y, m)
    assert(AL.shape == (1,X.shape[1]))
    
    return AL, caches

def retropropagacion_L_capas(AL, Y, caches):
    """
    Implementa la retropropagacion para el grupo [LINEAL->RELU] * (L-1) -> LINEAL -> SIGMOIDE
    
    Argumentos:
    AL -- vector de predicciones, salida de la propagación hacia delante (propagacion_L_capas())
    Y -- Vector de etiquetas verdaderas de dimensiones (1, número de casos de entrenamiento).
    caches -- lista del cache de cada capa. Contiene:
                cada cache linear_activation_forward("relu") (es caches[l], para l en range(L-1), es decir, l = 0...L-2)
                el cache de linear_activation_forward("sigmoide") (es caches[L-1])
    
    Devuelve:
    gradientes -- Un diccionario con los gradientes
                  gradientes["dA" + str(l)] = ... 
                  gradientes["dW" + str(l)] = ...
                  gradientes["db" + str(l)] = ... 
    """
    gradientes = {}
    # Numero de capas
    L = len(caches)
    m = AL.shape[1]
    # Tras esta linea, Y tiene las mismas dimensiones que AL.
    Y = Y.reshape(AL.shape)
    
    # Inicializar la retropropagacion
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    # Capa L: gradientes de (SIGMOID -> LINEAL). Entrada: "dAL, cache_actual". 
    #                                            Salidas: "gradientes["dAL-1"], gradientes["dWL"], gradientes["dbL"]
    # la capa L es el ultimo valor de la lista "caches", que contiene L elementos, y al empezar el indice por 0, el ultimo es L-1
    cache_actual = caches[L-1] # tambien cache_actual = caches[-1] nos lleva al ultimo elemento
    dA_prev_temp, dW_temp, db_temp = backward_activacion_lineal(dAL, cache_actual, activacion = "sigmoide")
    gradientes["dA" + str(L-1)] = dA_prev_temp
    gradientes["dW" + str(L)] = dW_temp
    gradientes["db" + str(L)] = db_temp
    
    # Bucle de l=L-2 a l=0 para implementar los gradientes de (RELU -> LINEAL)
    for l in reversed(range(L-1)):
        # Entradas: "gradientes["dA" + str(l + 1)], cache_actual". 
        # Salidas: "gradientes["dA" + str(l)] , gradientes["dW" + str(l + 1)] , gradientes["db" + str(l + 1)]
        # La suma (l + 1) es por el hecho de que nuestro indice va de (0, L-1) y no de (1, L), por lo que la equivalencia se 
        # logra sumando 1.
        cache_actual = caches[l]
        dA_prev_temp, dW_temp, db_temp = backward_activacion_lineal(gradientes["dA" + str(l + 1)], 
                                                                    cache_actual, 
                                                                    activacion = "relu")
        # Otra forma equivalente, que es lo que tenemos encapsulado en 2 lineas en la funcion backward_activacion_lineal():
        # dA_prev_temp, dW_temp, db_temp = backward_lineal(relu_backward(grads["dA" + str(l + 1)], current_cache[1]), current_cache[0])
        gradientes["dA" + str(l)] = dA_prev_temp
        gradientes["dW" + str(l + 1)] = dW_temp
        gradientes["db" + str(l + 1)] = db_temp        

    return gradientes

# Descomentar para debuggear

# def modelo_L_capas(X, Y, dims_capas, tasa_aprendizaje = 0.0075, num_iteraciones = 3000, dibujar_coste=False):
#     """
#     Implementa una red neuronal de L capas: [LINEAL->RELU]*(L-1)->LINEAL->SIGMOIDE.
    
#     Argumentos:
#     X -- datos de entrada, de dimensiones (n_x, numero de ejemplos)
#     Y -- vector de etiquetas verdaderas de dimensiones (1, numero de ejemplos)
#     dims_capas -- lista que contiene los tamaños de la capa de entrada y el resto de capas.
#                   Es decir, contiene L + 1 elementos, siendo L el numero de capas de la red 
#                   (solo se cuentan ocultas y la de salida)
#     tasa_aprendizaje -- tasa de aprendizaje de la regla de actualización de descenso de gradiente    
#     num_iteraciones -- numero de iteraciones del bucle de optimización
#     dibujar_coste -- si True, se imprimira el coste cada 100 iteraciones 
    
#     Devuelve:
#     parametros -- un diccionario que contiene los parametros hallados por el modelo; 
#                   pueden utilizarse para hacer predicciones.
#     """

#     np.random.seed(1)
#     # Lista para llevar registro de los costes
#     costes = []
    
#     # Inicializamos los parametros
#     parametros = inicializacion_profunda(dims_capas)
    
#     # Bucle que aplica el descenso del gradiente
#     for i in range(0, num_iteraciones):
        
#         # Forward propagation: [LINEAL -> RELU]*(L-1) -> LINEAL -> SIGMOIDE.
#         AL, caches = propagacion_L_capas(X, parametros)

#         # Calcular el coste
#         coste = calcular_coste(AL, Y)
    
#         # Backward propagation.
#         gradientes = retropropagacion_L_capas(AL, Y, caches)
 
#         # Actualizacion de parametros
#         parametros = actualizar_parametros(parametros, gradientes, tasa_aprendizaje)
                
#         # Mostrar el coste cada 100 iteraciones de entrenamiento
#         if dibujar_coste and i % 100 == 0 or i == num_iteraciones - 1:
#             print("Coste tras cada iteración %i: %f" %(i, np.squeeze(coste)))
#         if i % 100 == 0 or i == num_iteraciones:
#             costes.append(coste)
    
#     return parametros, costes

# train_set_x_orig, train_y, test_set_x_orig, test_y, clases = cargar_dataset()
# train_set_x_re = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
# test_set_x_re = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
# train_x = train_set_x_re / 255.
# test_x = test_set_x_re / 255.
# L_dims_capas = [12288, 20, 7, 5, 1]
# tasa_aprendizaje_L = 0.0075
# parametros_L, costes_L = modelo_L_capas(train_x, train_y, L_dims_capas, tasa_aprendizaje_L, num_iteraciones = 3000, dibujar_coste=True)