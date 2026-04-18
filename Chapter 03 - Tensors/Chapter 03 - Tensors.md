# Chapter 03 - Tensors

Hasta ahora, hemos conseguido crear estructuras como Unit, que nos permiten automatizar los procesos matemáticos que involucran al Deep Learning, pero no de la manera más eficiente.

Los frameworks más potentes como PyTorch o TensorFlow, no operan a nivel de un valor, como hemos hecho hasta ahora, sino que operan con una estructura de datos que permite aprovechar las propiedades de paralelización del hardware para hacer el entrenamiento e inferencia mucho más óptimos.

## 1. Tensores

En concreto, estos motores usan lo que denominamos tensores, que permiten guardar un array de valores contiguos en memoria, pudiendo adoptar diferentes formas:

- **Escalar**: Un tensor con un único valor, es equivalente a la clase Unit que hemos creado, ejemplo Tensor(1):

$$x = 42.5$$

- **Vector**: Un tensor compuesto por un array, cuyos valores se leen en línea recta, actúa igual que un vector matemático, ejemplo Tensor(3):

$$\mathbf{v} = \begin{bmatrix} 1.2 \\ -3.0 \\ 5.5 \end{bmatrix}$$

- **Matriz**: Un tensor cuya información se estructura en filas y columnas, pudiendo hacer operaciones matemáticas en forma de matrices, ejemplo Tensor(2,3):

$$\mathbf{M} = \begin{bmatrix} 1.0 & 2.5 & 3.1 \\ -1.2 & 0.0 & 4.8 \end{bmatrix}$$

- **Tensor**: Es un tensor con más de dos dimensiones, como un cubo o un hipercubo, por conveniencia, denominamos tensor a todas las formas anteriores, ejemplo Tensor(3,2,2):

$$ \mathcal{T}_{i,j,k} = \begin{bmatrix}
        \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}_1 &
        \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}_2 &
        \begin{bmatrix} 9 & 0 \\ 1 & 2 \end{bmatrix}_3
    \end{bmatrix} $$

Como vemos la forma o `shape` de un tensor se dispone en la forma de: **(dim_n, ... , batch, row, column)**. Próximamente veremos el porque de esta disposición.

### 1.1. Strides

Los valores de un tensor siempre se encuentran dispuestos de forma contigua dentro de la memoria, sin importar la forma que es este tenga, por ejemplo, si tenemos:

$$\mathbf{v} = \begin{bmatrix} 1.0 \\ 2.0 \\ 3.0 \\ 4.0 \end{bmatrix}, \mathbf{M} = \begin{bmatrix} 1.0 & 2.0 \\ 3.0 & 4.0 \end{bmatrix}$$

En memoria, la información de ambos tensores se verán en la forma de `vector<double> data  = { 1.0, 2.0, 3.0, 4.0 }`. Entonces, ¿cómo hacemos para recorrer la información de forma diferente según la forma del tensor?

Aquí es donde entra el concepto de `strides`, que nos indican cuantos valores tenemos que saltar en memoria para viajar de un valor a otro en una misma dimensión. Estos también siguen la forma **(dim_n, ... , batch, row, column)**.

Para nuestros ejemplos, los strides el tensor $v$ serán `vector<int> strides  = { 1.0 }`, sin embargo para el tensor $M$ tendremos `vector<int> strides = { 2.0 , 1.0 }`, ya que si queremos movernos por columnas, deberemos de movernos de uno en uno, pero si queremos saltar de fila en fila, deberemos de hacerlo de dos en dos. 

Los strides son vitales para realizar operaciones más complejas como la multiplicación de matrices, si queremos calcular $M \times M$, la segunda matriz deberá de recorrerse por filas (de dos en dos), usando los `strides`.

$$
\begin{bmatrix}
\colorbox{#FFF9C4}{$1.0$} & \colorbox{#FFF9C4}{$2.0$} \\
3.0 & 4.0
\end{bmatrix}
\times
\begin{bmatrix}
\colorbox{#C8E6C9}{$1.0$} & 2.0 \\
\colorbox{#C8E6C9}{$3.0$} & 4.0
\end{bmatrix}
=
\begin{bmatrix}
(1.0 \cdot 1.0 + 2.0 \cdot 3.0) & \dots \\
\dots & \dots
\end{bmatrix}
$$

De tal manera, podemos usar los `strides` para acceder a cualquier elemento de un tensor como si fueran coordenadas. Si queremos un valor en la coordenada $(i,j,k)$:

$$ physical\_index = (i \times strides_0) + (j \times strides_1) + (k \times strides_2) $$

## 2. Implementación de la clase Tensor

> Definición de la clase en `/include/tensor.h`

> Implementación de la clase en `/src/tensor.cpp`
### 2.1. Estructura base

A partir de todo lo que conocemos hasta ahora sobre los tensores, la implementación es bastante directa, aunque existen un par de trucos que hacen que el código sea mucho más simple.

Esta clase, contendrá de momento los siguientes atributos relativos a la forma del tensor:

- `shared_ptr<float[]> data`: Un puntero compartido hacia un array de numeros reales en memoria, contendrá los valores del tensor.

- `size_t total_size`: Es el número total de elementos de un tensor. Su tipo es `size_t`, que es un entero especial de C++ para medir el tamaño de un array. Nunca es negativo, y representa como máximo el tamaño del objeto más grande que se pueda almacenar en el sistema.

- `vector<int> shape`: Es un array de enteros que representan la forma de un tensor.

- `vector<int> strides`: Array de enteros que almacena los *stride* del tensor.

Al igual que la clase Unit, los tensores deben de tener la capacidad de crear un grafo de computación para realizar descenso por el gradiente. 

Para ello, el Tensor contendrá:

- `vector<shared_ptr<Tensor>> parents`: Una lista de punteros a los Tensores padres o previos que han creado al Tensor actual.
- `shared_ptr<Tensor> grad`: Si para Unit, una clase con un solo valor, su gradiente era un solo valor, para Tensor, el gradiente vendrá dado por otro Tensor, cuyos valores corresponden uno a uno con lo gradientes de cada valor del Tensor original.
- `function<void()> _backward`: Una función que indique como propagar el gradiente a través del Tensor.

### 2.2. Constructor

A partir de todo lo que conocemos ahora mismo, la implementación del constructor del Tensor es directa, aunque tiene algunos detalles a comentar.

El constructor contiene los siguiente parámetros:

- `vector<int>& shape_param`: Es la referencia a un vector de enteros con la forma que queremos que tenga el Tensor.
- `vector<float>& data_param`: Referencia a un vector de números reales con la `data` que contendrá el Tensor.
- `vector<shared_ptr<Tensor>>& parents_param`: Referencia a un vector de punteros compartidos que apuntan a los padre del Tensor.

**Inicialización de los datos:**

Para inicializar la `data` del tensor asignamos a este atributo del Tensor un puntero compartido a un array de `floats` en memoria del tamaño de `total_size` elementos.

La parte más crítica del constructor es la inicialización de los `strides`. Para ello podemos definir el proceso de creación de cualquier stride genérico como:

$$
\begin{cases} 
S_{n-1} = 1 \\
S_i = S_{i+1} \times d_{i+1}
\end{cases}
$$

Esta expresión nos dice que el `stride` de una dimensión es la multiplicación de las anteriores dimensiones comenzando por la derecha. Por ejemplo, para un tensor con forma Tensor(3,2,5):

- $ S_2 = 1 $
- $ S_1 = 5  $ (5 columnas)
- $ S_0 = 2 \times 5 = 10$ (5 columnas y 2 filas)

Como vemos, el `stride` de la última dimensión (las columnas), es siempre $1$, a esto le llamamos la "dimensión rápida", es decir, la dimensión cuyos valores están contiguos en memoria. Establecemos las columnas por convención para seguir la misma arquitectura que PyTorch, NumPy o TensorFlow. Existen otros entornos cuya dimensión rápida es la fila, como R o Matlab.

### 2.3. Actividades

Como actividad para reforzar estos conceptos, puedes comprender el código de las funciones de inicialización básicas: `ones`, `zeros` y `random`. Y analizar todo el flujo desde que se llama a estas funciones hasta que el tensor esté completamente inicializado en memoria.

Por otro lado, puedes ejecutar el test `00_test_tensor.cpp`, ejecutando dentro de la carpeta `src/`: 

```
g++ ../tests/00_test_tensor.cpp tensor.cpp utils.cpp  -I../include -o test_tensor && ./test_tensor
```

En él, podrás observar como se crean dos tensores, uno en forma de matriz y otro de tres dimensiones. Por consola aparecerán detalles informativos de cada tensor proporcionados por la función `info`, y verás como usando los strides podemos obtener cualquier dato del tensor en forma de coordenada. 

Intenta encontrar el último elemento de cada tensor, si te equivocas, por pantalla puede aparecer un número aleatorio como `-1.329e-7`, o simplemente un cero. Eso indicará que has intentado obtener un valor en memoria superior al tamaño del tensor, y estas mirando lo que hay en tu memoria en ese punto.


## 3. Bibliografía y Recursos Adicionales

**Vídeos y Cursos**

**Artículos y libros**
* [Documentación de la clase Tensor de PyTorch](https://docs.pytorch.org/docs/stable/tensors.html) - Documentación oficial de PyTorch acerca  de los Tensores

- [PyTorch Internals](https://blog.ezyang.com/2019/05/pytorch-internals/) - Artículo del progesor Edward Z. Yang, sobre como funciona PyTorch.

**Repositorios**
* [Repositorio de GitHub de PyTorch](https://github.com/pytorch/pytorch)

