# Chapter 01 - Automating backpropagation

Una vez comprendidas las matemáticas que hay detras del aprendizaje de las redes neuronales, vamos nuestro primer mini framework de redes neuronales, que nos permita automatizar la retropropagación del gradiente y entrenar nuestros primeros modelos de forma efectiva.

Este capitulo no pretende explicar el código línea por línea, sino exponer la lógica detrás del código y como este ha sido diseñado. Para aprender más sobre la implementación, en cada subapartado se indicarán las rutas con el código comentado para que pueda revisarlo junto con la explicación.

Destacar que este capítulo esta altamente inspirado en el repositorio de mircrograd de Andrej Karpathy, recomiendo echar un vistazo tanto a su código, como al vídeo que tiene haciéndolo desde cero, ambos enlaces en la bibliografía.

## 1. Clase Unit

> Definición de la clase en `/include/unit.h`

> Implementación de la clase en `/src/unit.cpp`

Unit es la clase más básica del framework **BlockTorch**, esta representa un único valor real, y nos permite realizar operaciones simples entre ellas como la suma, resta, multiplicación y división, creando paralelamente un **grafo de computación**, sobre el que podemos propagar el gradiente hacia detrás.

Por ejemplo si tenemos:
```
a = Unit(2.0)
b = Unit(3.0)
c = Unit(4.0)

d = a + b
e = d * c
```

Se generará automáticamente algo tal que:
<p align="center">
<img src=../images/01/computation_graph.svg>
<p>

Este grafo no solo almacena el valor de cada Unit, sino que también podremos saber el valor del gradiente actual de cada nodo y además podremos asignarle una función de propagación diferente según las operaciones implicadas. 

Por ello, nuestra clase Unit va a tener los siguientes atributos:

- `double data`: Es el valor real del Unit
- `double grad`: Es el valor del gradiente del Unit, inicializado normalmente a 0.0.
- `function<void()> backward`: Permite asignar una función que se ejecutará durante la retropropagación.

- `vector<shared_ptr<Unit>> children`: Almacena una lista de punteros en memoria a los hijos de este Unit, que serán otros Unit. Por ejemplo, para el Unit `d`, sus hijos serán `a` y `b`.

### 1.1. Función de retropropagación

La única función asociada a esta clase es la denominada `void retropropagate()`, que como su propio nombre indica, se encarga de inicializar la retropropagación del gradiente desde el nodo en el que se haya llamado.

Esta no recibe ningún parámetro, ya que su única tarea es ir hacia detrás viajando por los hijos hasta llegar al primero de ellos, con el objetivo de crear un órden topológico de los nodos, es decir, una lista con los nodos desde el primero que se creó hasta el último, para nuestro ejemplo será `[a,b,c,d,e]`.

Debemos de conocer esto, ya que es indispensable propagar el gradiente en el orden correcto, según se hayan creado los nodos, desde el último hasta el primero. Por ello, la función a continuación invierte este orden y llama a las funciones `backward` de cada Unit, estableciendo antes el gradiente del nodo actual a 1.0, para que al propagarse, no de todo 0.0.

## 2. Operaciones

> Definición de las operaciones en `/include/ops.h`

> Implementación de las operaciones en `/src/ops.cpp`

Ahora crearemos las funciones que nos permitan realizar operaciones entre Units. 

Cada operación esta dividida en dos, la primera parte, que es la más sencilla, consta de la creación del Unit que se pasará como respuesta, que no será más que un Unit con un valor proveniente de la operación realizada entre los dos Units de entrada, y asociándole estos como hijos.

En nuestro ejemplo, para el Unit `d`:

- data: `a->data + b->data`
- children: `[a,b]`

La segunda parte, se encarga de asignar una función `backward` al Unit de salida, que se encargará de propagar el gradiente a sus hijos. 

Según la operación tendremos:

- **Operación de suma:** La operación suma es la más sencilla de todas, si nosotros tenemos que:

  $$ d = a + b$$

  Las derivadas parciales tanto de $a$ como $b$ serán:

  $$ \frac{\partial d}{\partial a} = 1 , \frac{\partial d}{\partial b} = 1 $$
   
  Por tanto, para propagar el gradiente debemos de seguir la regla de la cadena, y multiplicar a esto el gradiente de $d$, quedando las siguientes derivadas locales:

  $$ \frac{\partial d}{\partial a} = 1 \cdot  \frac{\partial L}{\partial d}, \frac{\partial d}{\partial b} = 1 \cdot  \frac{\partial L}{\partial d} $$

  Es literalmente, acceder al gradiente actual de $d$ y pasarselo a sus hijos.

- **Operación de multiplicación**: Si tenemos según el ejemplo anterior:
  $$ e = c \cdot d $$

  Las derivadas parciales tanto de $c$ como $d$ serán:
  $$ \frac{\partial e}{\partial c} = d, \frac{\partial e}{\partial d} = c $$

  Seguimos el mismo proceso de la regla de la cadena y tenemos:

    $$ \frac{\partial e}{\partial c} = d \cdot  \frac{\partial L}{\partial e}, \frac{\partial e}{\partial d} = c \cdot  \frac{\partial L}{\partial e} $$

  Para definir la función de backward, cogeremos el gradiente del Unit de salida, y lo multiplicaremos por $d$ para el gradiente $c$ y viceversa para el gradiente de $d$.

- **Operación de resta**: Casi idéntica a la suma, el único cambio sería que la derivada parcial del sustraendo tiene valor $-1$ en lugar de $1$, por ende, debemos de pasar el gradiente del resultado con un signo negativo.

- **Operación de elevado:** Si tenemos un Unit $a$ elevado a una constante $b$:

  $$c = a^b$$

  La regla matemática de la derivación para una potencia nos dice que bajamos el exponente y le restamos uno a la base:

  $$\frac{\partial c}{\partial a} = b \cdot a^{b-1}$$

  Por tanto, al aplicar nuestra querida regla de la cadena multiplicando por el gradiente que viene de atrás, nos queda:

  $$\frac{\partial L}{\partial a} = b \cdot a^{b-1} \cdot \frac{\partial L}{\partial c}$$

  En nuestra función `backward`, calculamos exactamente esto usando la función `pow` de la librería estándar de C++ y lo multiplicamos por `out->grad`.

- **Operación de división:** Para simplificar esta operación al máximo y no tener que programar la tediosa regla de la cadena, vamos a usar un truco matemático. Dividir $a$ entre $b$ es exactamente lo mismo que multiplicar $a$ por $b^{-1}$:

  $$c = \frac{a}{b} = a \cdot b^{-1}$$

  Como ya tenemos programadas de manera independiente las operaciones de multiplicación (`*`) y de potencia (`pow`), ¡no necesitamos escribir una función `backward` nueva! Simplemente reutilizamos nuestras funciones llamándolas directamente y el grafo de computación hará el resto del trabajo por nosotros.

- **Operación ReLU (Función de activación):** Las funciones de activación se usan para introducir no linealidad en las redes. ReLU es la más famosa y sencilla. Si el valor de entrada es menor que 0, lo corta y devuelve 0; si es mayor, lo deja pasar tal cual:

  $$b = \max(0, a)$$

  Su derivada es igual de simple. Si el valor de salida fue mayor que 0, la pendiente (derivada local) es $1$. Si el valor fue cortado a 0, la pendiente es $0$.

  $$\frac{\partial b}{\partial a} = \begin{cases} 1 & \text{si } a > 0 \\ 0 & \text{si } a \le 0 \end{cases}$$
  
  Aplicando la regla de la cadena en el código, si `out->data > 0`, propagamos el gradiente tal cual (`1 * out->grad`); en caso contrario, detenemos el gradiente sumando `0.0`.

- **Operación Tanh (Función de activación):** Otra función matemática muy útil que "aplasta" cualquier número para que encaje en un rango entre -1 y 1. 

  $$b = \tanh(a)$$

  Por definición matemática, la derivada de la tangente hiperbólica es uno menos ella misma al cuadrado:

  $$\frac{\partial b}{\partial a} = 1 - \tanh^2(a)$$

  En nuestra función `backward` evaluamos esta derivada local y, como siempre, la multiplicamos por el gradiente de salida `out->grad` para completar la retropropagación.

> En el archivo `/tests/01_basic_computation_graph.cpp.h` podrás encontrar un pequeño código con la creación del grafo computacional de este ejemplo, y como al llamar a la función `retropropagate()`, el gradiente se propaga correctamente a todos sus nodos.

Linux (en `/src`):
```
g++ ../tests/00_basic_computation_graph.cpp ops.cpp unit.cpp -I../include -o test_basic_computation_graph && ./test_basic_computation_graph
```

Windows (en `/src`):
```
g++ ../tests/00_basic_computation_graph.cpp ops.cpp unit.cpp -I../include -o test_basic_computation_graph.exe && .\test_basic_computation_graph.exe
```

## 3. Detalles del código y gestión de memoria

Si nos fijamos en el código, tanto de la clase `Unit` como de las operaciones, encontramos un elemento esencial para la arquitectura de nuestro framework: los punteros inteligentes `std::shared_ptr`. En BlockTorch, todos los nodos están envueltos en ellos.

Estos son una evolución de los clásicos punteros de memoria de C++, con la gran ventaja de que implementan un concepto llamado **"propiedad compartida" (shared ownership)** y se autogestionan solos.

En un grafo computacional de una red neuronal, un mismo nodo inicial puede ser utilizado por múltiples operaciones matemáticas simultáneamente. Por ejemplo, un `Unit` llamado $a$ podría ser hijo tanto de una suma como de una multiplicación al mismo tiempo. 

¿Qué significa esto a nivel de código? 
Que múltiples variables y vectores (como las listas de `children` de las operaciones) van a "apuntar" a la misma dirección de memoria de $a$. El `shared_ptr` lleva un contador interno automático de cuántas partes de nuestro código están referenciando a ese Unit. En el momento en el que ese Unit deja de ser necesario para el grafo y su contador llega a cero, el puntero libera la memoria automáticamente. 

Esto nos permite construir arquitecturas complejas sin tener que destruir manualmente la memoria usando `delete`, evitando por completo los temidos *memory leaks* (fugas de memoria) y manteniendo el código limpio y seguro.

Toda ventaja tiene su inconveniente, el precio de estos punteros es que son más lentos que los clásicos, por ellos frameworks como PyTorch usan sus propias clases de punteros personalizados para obtener más rendimiento.

## 4. Bibliografía y Recursos Adicionales

**Vídeos y Cursos**
* [Lecture 4 | Introduction to Neural Networks ](https://www.youtube.com/watch?v=d14TUNcbn1k) - Standfor University School of Engineering. Lectura oficial de la Universidad de Standfor sobre redes neuronales y grafos computacionales.
* [The spelled-out intro to neural networks and backpropagation: building micrograd](https://youtu.be/VMj-3S1tku0) - Andrej Karpathy. *(Inspiración principal para la construcción del código de este capítulo).*

**Repositorios**
* [Micrograd (GitHub)](https://github.com/karpathy/micrograd) - Repositorio oficial de Andrej Karpathy con la implementación original en Python.

**Artículos y Documentación**
* [Official C++ documentation about shared_ptr](https://en.cppreference.com/w/cpp/memory/shared_ptr.html) - Documentación oficial de C++ sobre los punteros compartidos.

* [Calculus on Computational Graphs: Backpropagation](http://colah.github.io/posts/2015-08-Backprop/) - Artículo muy visual de cómo funcionan los grafos computacionales y como propagar el gradiente.