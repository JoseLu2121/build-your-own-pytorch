# Chapter 02 - Your first neuronal net

En este capitulo construiras tu primera red neuronal, usando el código de la anterior lección.

Nos adrentaremos en que operaciones realizan las neuronas y que es un perceptrón multicapa o MLP, a partir de ahí, crearemos nuevas clases que, utilizando Units, se comporten como redes neuronales y las entrenaremos con un pequeño dataset.

Por ello, primero debemos de hacernos la pregunta de, ¿qué es una neurona artificial?.

## 1. ¿Qué es una neurona en Deep Learning?

Dentro de la disciplina del Deep Learning, el concepto más popular es el de neurona artificial, inspiradas en las neuronas biológicas.

Estas neuronas no tienen nada de especial, son sencillamente una función matemática de una o varias entradas y una única salida. 

Esta función contiene tres elementos fundamentales:

- **Pesos:** Son los parámetros optimizables de las neuronas. Cada uno se multiplica por una de las entradas de la neurona, ponderando las entradas, otorgando más importancia a una o a otra dependiendo de los datos de entrenamiento.

- **Sesgo o Bias**: A la suma de todos los pesos multiplicados por las entradas se le suma un único valor llamado bias o sesgo, que actúa como un umbral, determinando cuando la neurona se "activa".

- **Funciones de activación**: A todo el resultado anterior se le aplica una función matemática no lineal, que determina el comportamiento de la neurona. Por ejemplo si esta aplasta o acota el resultado, como la función `relu`, una de las más usadas, que acota el resultado a solo valores positivos.

Matemáticamente una neurona se describe como:

$$y = f \left( \sum_{i=1}^{n} w_i \cdot x_i + b \right)$$

Siendo: 

- $w_i$: Los pesos o parámetros entrenables.
- $x_i$: Entradas de la neurona.
- $b$: Bias o sesgo.
- $f$: Función de activación.
- $y$: Salida de la neurona.

<p align="center">
<img src=../images/02/neuron_diagram.svg>
<p>

### 1.1. Implementación usando Units

> Definición de la clase en `/include/neuron.h`

> Implementación de la clase en `/src/neuron.cpp`

Para implementar una neurona usando Units, tendremos que creaer una clase `Neuron` que contenga dos atributos:
- `vector<shared_ptr<Unit>> weights`: Son Units que representan los pesos de nuestra neurona, habrá siempre uno por entrada, y se iniciarán con valores aleatorios para un correcto entrenamiento.

- `shared_ptr<Unit> bias`: Un único Unit que represente a nuestro sesgo o bias, que se inicializará en nuestro caso siempre a 1.0, aunque lo común es que sea un nuevo aleatorio entre -1.0 y 1.0.

Esta clase tendrá dos funciones muy básicas:

- `forward`: Computa la neurona con las entradas que le pasemos como parámetros, multiplicandolas por los pesos una a una, sumándole el sesgo, y finalmente se retorna el resultado pasado por una función de activación, en nuestro caso siempre será la tangente hiperbólica, pero en un futuro tendremos que crear arquitecturas que nos permitan tener diferentes funciones de activación en cualquier momento.

- `parameters`: Simplemente retorna los parámetros entrenables de la neurona para poder optimizarlos en el entrenamiento, en nuestro caso retorna un vector de Units con los pesos y el bias.

## 2. Capas de neuronas

Una única neurona es capaz de resolver problemas extremadamente sencillos, pero si queremos poder resolver problemas más complejos, debemos de agrupar varias neuronas que acepten los mismos datos de entrada. Esto se denomina capa de neuronas o *layer* en inglés.

Por ejemplo, si tenemos una capa con tres neuronas, y le pasamos 2 datos de entrada, cada una de ellas tendrá 2 pesos y un sesgo completamente independientes de las demás neuronas. La capa devolverá 3 números asociados a la salida de cada neurona.

<p align="center">
<img src=../images/02/layer_diagram.svg>
<p>

### 3.1. Implementación de una capa
> Definición de la clase en `/include/layer.h`

> Implementación de la clase en `/src/layer.cpp`

Para implementar una capa de neuronas en nuestro mini-framework de Units, crearemos una clase `Layer` con un único atributo:

`vector<std::shared_ptr<Neuron>> neurons`: Es una lista de punteros hacia instancias de clases `Neuron` del subapartado anterior.

Para inicializar una capa, tendremos un constructor con dos parámetros: `inputs` (entradas) y `out` (salidas). Bastará con crear tantas neuronas como salidas tengamos, y cada neurona con las entradas pasadas por parámetro.

Al igual que la clase `Neuron`, `Layer` tendrá las mismas dos funciones:

- `forward`: Computa la capa llamando a las funciones `forward` de cada neurona con las entradas pasadas por parámetro.

- `parameters`: Devuelve todos los parámetros entrenables de la capa, de nuevo, llamaremos a esta misma función de las neuronas y devolveremos un vector con todos los parámetros.

## 3. Perceptrón Multicapa (MLP)

Es la estructura más clásica y simple de las redes profundas, es decir, redes neuronales de más de una capa. 

En esta estructura, tenemos varias capas apiladas horizontalmente, de manera que cada entrada de una capa es la salida de la anterior.

Es muy común el término de `dense layers` o capas densas, es decir, cada salida de cada neurona, está conectada a todas las entradas de la siguiente capa.

<p align="center">
<img src=../images/02/mlp_diagram.svg>
<p>

Además, esta estructrura se puede dividir en:

- **Input layer**: Es la capa de entrada del perceptrón, recibe los datos de entrada en crudo.

- **Hidden layer**: Las capas ocultas son aquellas que reciben como entrada la salida de la capa anterior, puede haber más de una.

- **Output layer**: Es la última capa del perceptrón y devuelve la salida final de esta.

### 3.1. Implementación del perceptrón

> Definición de la clase en `/include/mlp.h`

> Implementación de la clase en `/src/mlp.cpp`

La implementación de un perceptrón se hará mediante una clase `MLP` con un único atributo:

- `vector<shared_ptr<Layer>> created_layers`: Es una lista de punteros hacia las capas del perceptrón.

De esta manera nuestro perceptrón tendrá un constructor que acepte como parámetros:

- `int inputs`: Cantidad de inputs que aceptará el perceptrón.

- `vector<int> outputs`: Es una lista con la cantidad de salidas que queramos que tengan cada una de las capas del perceptrón.

Con esta información, somos capaces de crear cada una de las capas que componen el MLP, por ejemplo, para crear un perceptrón como el del diagrama anterior, nuestro parámetro `inputs` valdrá 1, ya que solo acepta un input por neurona, por otro lado, el parámetro `outputs` será un vector con los valores [3,2,1]. A partir de aquí llamaremos al constructor de `Layer` con las dimensiones adecuadas para cada capa.


Las funciones definidas para esta clase son:

- `forward`: Al igual que para `Neuron` y `Layer`, computa el perceptrón, llamando a las funciones correspondientes de cada capa, y pasando de una a la entrada de la siguiente.

- `parameters`: Devuelve de una todos los parámetros entrenables de todo el perceptrón.

- `zero_grad`: Establece todos los gradientes de los parámetros entrenables a 0.0. Esto es esencial para un entrenamiento correcto, ya que si no hacemos esto antes de cada iteración, los gradientes se acumularían y daríamos saltos gigantescos a la hora de optimizar los pesos.

- `fit`: Entrena el perceptrón sobre un dataset durante un cierto número de iteraciones. Como entrada recibe el dataset, dividido en `inputs`, es decir, las variables independiente, y en `targets`, una lista con las variables objetivos que queremos que devuelva nuestro perceptrón para cada entrada, es decir nuestra $y_{target}$ del primer capítulo. También recibe el número de iteraciones `iterations` y un learning rate `learning_rate` explicado en el capítulo 00.

  La función por cada iteración computa el perceptrón hacia delante llamando a `forward` con cada entrada de `inputs`, para después generar un Unit que represente la función $loss = (y - y_{target})²$.

  Con esta información generamos un Unit con la media de estos errores. A partir de este Unit propagaremos el gradiente y aplicaremos descenso por el gradiente para optimizar cada uno de los parámetros entrenables.


## 4. Entrenamiento de una red neuronal

Usaremos el código creado en este capítulo para entrenar nuestra primera red neuronal en forma de perceptrón multicapa.

En concreto, resolveremos uno de los problemas que impulsó el desarrollo de los perceptrones multicapa, el problema del XOR. Al ser un problema no lineal, una sola neurona no nos permite resolverlo, pero varias capas dispuestas una detrás de otra sí.

El problema consiste en que una red aprenda a calcular la operación de XOR por sí sola, para ello crearemos un dataset con todos lo valores posibles `inputs` y los resultados que da la operación con esos valores `targets`:

- XOR(0,0) = 0
- XOR(0,1) = 1
- XOR(1,1) = 0
- XOR(1,0) = 1

Entrenaremos al perceptrón durante 10000 iteraciones sobre este dataset, con un learning rate de 0.01. 
Al ejecutar en `src/`:

Linux
```
g++ ../tests/01_xor_problem.cpp unit.cpp ops.cpp mlp.cpp layer.cpp neuron.cpp -I../include -o test_xor && ./test_xor
```

Windows
```
g++ ../tests/01_xor_problem.cpp unit.cpp ops.cpp mlp.cpp layer.cpp neuron.cpp -I../include -o test_xor.exe && .\test_xor.exe
```

Verás como en cada iteración los resultados se acercan cada vez más a los esperados hasta obtener valores muyr cercanos. ¡Has entrenado tu primera red neuronal real!

## 5. Siguientes pasos

Con estos primeros tres capítulos del curso `build-your-own-pytorch`, has podido comprender las claves de todo framework de Deep Learning. Pero estos, no operan a nivel de un único valor como el Unit, sino que usan estructuras más complejas que permiten realizar estos cálculos de forma mucho más óptima, aprovechando la paralelización que nos aportan las CPUs y GPUs.

En los siguientes capítulos construiremos un framework mucho más real y cercano a como funcionan los más usados de la industria, como PyTorch o Tensorflow.


## 6. Bibliografía y Recursos Adicionales

**Vídeos y Cursos**
* [Lecture 4 | Introduction to Neural Networks ](https://www.youtube.com/watch?v=d14TUNcbn1k) - Standfor University School of Engineering. Lectura oficial de la Universidad de Standfor sobre redes neuronales y grafos computacionales.
* [The spelled-out intro to neural networks and backpropagation: building micrograd](https://youtu.be/VMj-3S1tku0) - Andrej Karpathy. *(Inspiración principal para la construcción del código de este capítulo).*

**Artículos y libros**

* [Deep Learning, Chapter 6: Deep FeedForward Networks](https://www.deeplearningbook.org/contents/mlp.html) Capítulo 6 de uno de los libros más populares sobre Deep Learning, usado para crear el ejemplo del problema XOR.

**Repositorios**
* [Micrograd (GitHub)](https://github.com/karpathy/micrograd) - Repositorio oficial de Andrej Karpathy con la implementación original en Python.

