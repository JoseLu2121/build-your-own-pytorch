# Chapter 00 - Basics

En el primer capitulo de este curso se explica la base matemática que hay detrás del Deep Learning, tocando a fondo todos los conceptos, desde los más básicos como, qué es una derivada, a los más complejos como el descenso por el gradiente o la retropropagación, con ejemplos incluidos de entrenamiento de neuronas manualmente.

## 1. Derivada

La derivada es uno de los conceptos matemáticos más importantes del Deep Learning, ya que nos permite optimizar los parámetros que hacen que las redes neuronales "aprendan".

### 1.1. Definición de derivada
La derivada de una función **$f$** en un punto **$x$** mide cuánto cambia la función f si modificamos la variable **$x$**.

Matemáticamente se define como: 

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

siendo h un valor muy pequeño.

Pongamos que tenemos la función **$f(x) = 2x + 3$**, si quisieramos conocer su derivada para el punto **$x = 2$**,

$$f'(2) =  \frac{f(2+0.0001) - f(2)}{0.0001}$$

lo que nos daría $f'(2) = 2$, que equivale a usando las reglas de derivación: $f'(x) = 2$. Lo que significa que para todo $x$, cada vez que incrementamos el valor de $x$ una unidad, $f(x)$, aumenta 2 unidades.

### 1.2. Derivada parcial
La derivada parcial mide cuanto cambia una expresión en función a una variable, por ejemplo, si tenemos la expresión:

$$ f(x,y) = x - y $$

La derivada parcial de **$x$** expresada como $\frac{\partial f}{\partial x}$, mide como cambia la función si solo modificamos el parámetro **$x$**, dejando **$y$** como una constante.

Esto es especialmente útil ya que podemos saber como modificar los pesos de una red neuronal, para que nuestra función de pérdida sea mínima. 

### 1.3. Caso práctico con una neurona

Pongamos que tengamos una red neuronal que consta únicamente de una sola neurona con una sola entrada y un solo peso:

- Peso: **$w = 1$**
- Entrada: **$input = 2$**

<p align="center">
<img src=../images/00/single_neuron.svg>
<p>


La idea es que nuestra salida sea $y_{target} = 10$, pero si nosotros multiplicamos la entrada por el peso tenemos:
**$$y = input \cdot w = 2$$**

Entonces, ¿cómo conseguimos que nuestra salida sea 10?, aquí es donde entran las derivadas parciales, entrenaremos esta neurona a mano desde cero. 

Primero, definiremos una función de pérdida como:
$$ loss = (y - y_{target})² $$

En nuestro caso 
$$ loss = ((input \cdot w) - y_{target} )² $$
$$ loss = (2 - 10)² = 64 $$

Como vemos el elevado a dos nos permite que el error siempre sea un número positivo. Por tanto, lo que querremos es que este error sea el menor posible, para ello debemos de conocer como varía este según modifiquemos el peso $w$.

Por consiguiente, calcularemos la derivada parcial, de $w$ con respecto a la función loss, denominada como $L$ en la siguiente expresión:  $\frac{\partial L}{\partial w}$.

Usaremos la expresión descrita al comienzo de este capítulo:
$$L(w) = (2*w - 10)²$$

$$L'(w) = \lim_{h \to 0} \frac{L(w+h) - L(w)}{h}$$

$$L'(1) = \lim_{h \to 0} \frac{L(1+0.0001) - L(1)}{0.0001} \approx \frac{63.9968004 - 64}{0.0001} \approx -32$$

De esta manera sabemos que la derivada parcial o gradiente del peso $w$ actualmente es -32. Lo que significa que si aumentamos el peso únicamente una unidad, el error disminuye dicho valor. Justamente lo que necesitamos, por lo que actualizaremos nuestro peso siguiendo la siguiente expresión:

$$ w = w - (0.1 \cdot -32) $$

$$ w = 4.2 $$

Se puede apreciar que multiplicamos el gradiente por un valor pequeño, este es el famoso **learning rate**, que mide como de grande es el paso que tomamos al actualizar los pesos, es decir, representa la velocidad de nuestro entrenamiento. Más adelante se explicará este concepto más detalladamente.

Si computamos de nuevo el $input$ por el nuevo peso $w$ y calculamos la función de error

$$ y = 2 \cdot 4.2 = 8.4 $$
$$ loss = (8.4 - 10)² = 2.56 $$

Como podemos observar nuestra predicción se acerca mucho más a $y_{target}=10$ y por consiguiente el error ha disminuido. 
¡Felicidades acabas de entrenar tu primera neurona!

### 1.4. Descenso por el gradiente

Al proceso que acabamos de realizar se le conoce como **descenso por el gradiente**. Es similar a estar con los ojos vendados en la cima de una montaña, e intentar descenderla únicamente conociendo la pendiente que sientes con tus pies.

Justamente esto es lo que hemos hecho al entrenar la neurona, pero en su lugar, la montaña es la función de error o pérdida, la persona vendada son los pesos y la pendiente es la derivada parcial o gradiente de los pesos con respecto a la función de pérdida. 

Matemáticamente tiene la siguiente expresión:
$$w = w - \alpha \cdot \frac{\partial L}{\partial w}$$

<p align="center">
<img src=../images/00/gradient_descent.svg>
<p>

El signo negativo nos sirve para movernos al lado contrario que el gradiente, ya que si este es positivo, la pendiente es hacia arriba, por lo que querremos "darnos la vuelta" y movernos hacia el lado contrario. 

Por otro lado $\alpha$ representa tal y como explicamos anteriormente, como de grande es el "paso" que damos cada vez que actualizamos el peso, denominado learning rate.

Nuestra intuición puede decirnos que usemos un learning rate lo mayor posible, ya que nuestra red neuronal aprenderá más rápido. Y esto en parte es así, si damos un "paso" mayor, llegaremos antes al valor mínimo del error. Pero, debemos de tener cuidado, ya que si nos movemos en intervalos mayores, corremos el riesgo de saltarnos el fondo del "valle" y rebotar en un lugar más "alto". Por ejemplo, en la gráfica anterior, si el último paso hubiera sido más largo, hubieramos acabado con un valor del error mucho mayor. 

Ajustar correctamente este parámetro es clave si queremos tener un entrenamiento estable, con una bajada constante del error en cada iteración.

En el archivo ```/src/01_single_neuron.cpp``` dentro de la subcarpeta de este capítulo, puedes encontrar un codigo sencillo con el ejemplo de pequeño entrenamiento que acabamos de realizar.

Para ejecutarlo basta con abrir una terminal dentro de la carpeta ```/src``` y ejecutar:

**Linux**: 

```
g++ 01_single_neuron.cpp -o single_neuron && ./single_neuron
```

**Windows**:
```
g++ 01_single_neuron.cpp -o single_neuron.exe && .\single_neuron.exe
```


### 1.5. Regla de la cadena

Una vez que ya sabemos como actualizar un parámetro para disminuir una función de coste mediante el uso de su derivada parcial, debemos de conocer en más profundidad como calcular gradientes de forma más eficiente. 

Supongamos que tenemos una red neuronal muy profunda con millones de de parámetros entrenables. La función que describe su comportamiento ya no depende solo de una entrada multiplicada por un peso, sino de millones de ellos, y con varias capas de operaciones más complejas hasta llegar a la función de coste. 

Por ende, calcular derivadas en base a su definición tal y como hemos hecho hasta ahora sería computacionalmente muy complejo, además, ¿cómo calculamos derivadas parciales de los pesos situados en la primera capa de nuestra red, si estas no estan conectadas directamente a la salida?

Aquí es donde entra el concepto de, la regla de la cadena o "chain-rule" en inglés. Esta establece que si tenemos varias operaciones encadenadas, podemos calcular cualquier derivada multiplicando las derivadas parciales desde el final hasta llegar a dicho parámetro.

Matemáticamente, si tenemos que $z$ depende de $y$, y a su vez $y$ depende de $x$, la derivada de $z$ con respecto a $x$ es:

$$\frac{\partial z}{\partial x} = \frac{\partial z}{\partial y} \cdot \frac{\partial y}{\partial x}$$

Hemos multiplicado el paso intermedio el gradiente del paso intermedio $y$ por el gradiente local de $y$ con respecto a $x$.

#### 1.5.1. Regla de la cadena en una Red Neuronal (Backpropagation)

Imaginemos que ampliamos nuestra red neuronal anterior y le añadimos una neurona más justo antes.
Esta neurona oculta (se denomina así por que no está conectada a la salida), no solo multiplica su entrada por un peso $w_1$, sino que le suma un valor que denominaremos sesgo o $bias$.

- Neurona oculta: Recibe un $input$, lo multiplica por su $w_1$ y le suma el bias $b$, a este resultado le denominaremos $h$: 

$$ h = input \cdot w_1 + b $$

- Neurona de salida: Recibe el resultado anterior $h$ y genera la salida $y$, multiplicandolo por un peso $w_2$:

$$ y = h \cdot w_2 $$

<p align="center">
<img src=../images/00/two_neurons.svg>
<p>

Finalmente, tenemos de nuevo una función de pérdida dada por:

$$ loss = (y - y_{target})² $$

Pongamos que tengamos los siguientes valores iniciales:
- $input = 2$
- $w_1 = 3$
- $b = 1$
- $w_2 = 2$
- $y_{target} = 10$

Computamos la pequeña red (forward pass):
- Neurona oculta: $ h = 2 \cdot 3 + 1 = 7 $
- Neurona de salida: $ y = 7 \cdot 2 = 14 $
- Error: $loss = (14 - 10)² = 16 $

Aqui nos surge el problema explicado previamente, como calculamos el gradiente del peso $w_1$ si nuestro error no depende directamente de él. Pues lo haremos propagando el gradiente hacia atrás utilizando la regla de la cadena, a esto se le conoce como retropropagación o backpropagation es inglés.

El gradiente de $w_1$ vendrá dado por:

$$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial h} \cdot \frac{\partial h}{\partial w_1}$$

Vamos desde la última operación hasta la primera propagando el gradiente:

$$ \frac{\partial L}{\partial y} = 2 \cdot (14 - 10) \cdot 1 = 8 $$

$$ \frac{\partial y}{\partial h} = 8 \cdot 2 = 16  $$

$$ \frac{\partial h}{\partial w_1} = 16 \cdot 2 = 32$$

Por tanto, si incrementamos nuestro peso $w_1$, el error aumentará considerablemente.



## 2. Siguientes pasos

Con esto, ya dispones de las bases matemáticas suficientes para entrenar como funcionan las redes neuronales por dentro, como entrenarlas propagando el gradiente hacia detrás y usando este para actualizar los pesos usando la técnica del descenso por el gradiente.

En el siguiente capítulo veremos como crear una clase en C++ que nos permita automatizar la retropropagación para no tenerla que calcular a mano, y entrenaremos nuestras primeras redes neuronales usando esa misma clase.

## 3. Bibliografía y Recursos Adicionales

Para la elaboración de este capítulo y para aquellos que quieran profundizar más en las matemáticas del Deep Learning, se recomiendan los siguientes recursos:

**Vídeos y Cursos**
* [Neural Networks, by 3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) - Una serie visual espectacular para entender la intuición matemática detrás de las redes neuronales.

**Artículos y Documentación**
* [CS231n: Derivatives, Backpropagation, and Vectorization](https://cs231n.stanford.edu/handouts/derivatives.pdf) - Apuntes oficiales de Stanford sobre el cálculo de gradientes y la regla de la cadena.







