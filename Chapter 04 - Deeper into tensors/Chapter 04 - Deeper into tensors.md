# Chapter 04 - Deeper into tensors

Ahora que tenemos una idea general de lo que son los tensores y conocemos algunas de sus propiedades como los `strides`, vamos a profundizar más en ellos, adentrándonos en uno de los conceptos más importantes de cualquier framework para poder realizar operaciones eficientemente: las vistas.

## 1. Vistas

Las vistas son indispensables para poder implementar operaciones entre tensores de manera eficiente y de forma genérica. Porque, ¿qué pasa si queremos sumar a una matriz un vector? ¿o un escalar?. Y, ¿cómo hacemos para convertir un tensor de tres dimensiones, en uno de sólo dos?

Para resolver estos problemas, usamos las **vistas**. Como vimos en el anterior capítulo, la forma del tensor es completamente independiente a los datos almacenados en memoria, estos siempre se disponen de forma contigua en esta, sin importar las dimensiones de estos. Por ello, una vista no es más que una "copia" de un tensor, cuyo puntero de datos apunta al mismo bloque de memoria que el del tensor original, pero este puede tener distinta forma y no afecta al grafo computacional.

<p align="center">
<img src=../images/04/view_diagram.svg>
<p>

### 1.1. Implementación

> Definición de las operaciones en `/src/tensor.h`

> Implementación de las operaciones en `/src/tensor.cpp`

Para implementar esto a nivel de código, crearemos un constructor de copia que acepte un tensor existente. Este constructor duplicará sus propiedades estructurales (`shape`, `strides` y `total_size`), y apuntará al mismo bloque de datos en memoria. Sin embargo, inicializaremos la lista de nodos padres (`parents`) vacía y el tensor de gradientes (`grad`) a `nullptr`. Esto desconecta de forma segura la vista del grafo computacional original, evitando efectos secundarios o dependencias indeseadas mientras manipulamos sus dimensiones.

La enorme utilidad de las vistas en LearnTorch reside en la abstracción y generalización del código. Tomemos como ejemplo la multiplicación de matrices (`matmul`). En lugar de programar múltiples versiones de la misma función para manejar tensores de 2, 3 o más dimensiones, usamos las vistas para convertir virtualmente cualquier tensor a exactamente 3 dimensiones `(Batch, Filas, Columnas)`. De este modo, nos basta con escribir un único algoritmo centralizado, introduciendo nuestras vistas en él y reduciendo así la complejidad de implementación drásticamente.

Por ello, dentro de `tensor.cpp`, podrás ver funciones como `view_to_3d` o `view_to_gemm` para poder crear dichas vistas adecuadamente. Las vistas también nos permiten crear una operación como `reshape`, muy popular en PyTorch, que permite modificar la forma de un tensor siempre que esta sea compatible con los datos en memoria.

## 2. Broadcasting

Al principio de este capítulo nos hemos hecho la pregunta de, ¿qué pasa si queremos sumar una matriz y un vector? Matemáticamente, sumar un tensor de forma (3, 3) con uno de forma (3) no es posible directamente. Sin embargo, en la práctica lo hacemos constantemente. La magia que lo permite se llama Broadcasting.

Técnicamente hablando, hacer broadcasting no es más que crear una vista muy específica y astuta de un tensor. Cuando aplicamos broadcasting, "expandimos" virtualmente el tensor más pequeño para que coincida con la forma del tensor más grande. La forma ineficiente de hacer esto sería reservar nueva memoria y copiar los datos del vector tantas veces como fuera necesario para construir una matriz del mismo tamaño. Sin embargo, usando nuestro conocimiento sobre las vistas y los strides, podemos lograr esto con cero copias en memoria.

<p align="center">
<img src=../images/04/tensor_sum.svg>
<p>

### 2.1. Uso de los strides

El secreto del broadcasting se encuentra en los strides, en concreto de asignarlos a valor $0$.

Supongamos que tenemos un vector de 3 elementos `[1, 2, 3]`. En memoria, esto está dispuesto de forma contigua.
* Su `shape` original es `(3)`.
* Su `stride` original es `(1)` (para avanzar al siguiente elemento, damos 1 salto en memoria).

Si queremos sumarlo a una matriz de `(3, 3)`, necesitamos que nuestro vector actúe como si también fuera una matriz de `(3, 3)`. Para lograrlo, creamos una **vista** de este vector con la siguiente configuración:
* Nueva `shape`: `(3, 3)`.
* Nuevos `strides`: `(0, 1)`.

¿Qué ocurre internamente cuando el motor de operaciones lee esta vista? 
Cuando lee las columnas de una fila, usa el stride `1`, leyendo `1, 2, 3`. Pero cuando salta a la siguiente fila (la primera dimensión), usa el stride `0`. Esto significa que **avanza 0 posiciones en memoria**, obligando al puntero a volver a leer exactamente los mismos tres datos físicos una y otra vez. 

<p align="center">
<img src=../images/04/strides_diagram.svg>
<p>


Hemos convertido un vector en una matriz visualmente repetida, ocupando exactamente la misma cantidad de RAM que el vector original, y sin tener que reasignar memoria en tiempo de ejecución, lo cuál sería muy ineficiente.

### 2.2. Implementación generalizada

> Definición de las operaciones en `/src/tensor.h`

> Implementación de las operaciones en `/src/tensor.cpp`

A nivel de código, LearnTorch dispone de una función (`broadcast_shapes`) que permite calcular el tensor resultante para hacer una operación entre dos tensores con `shape` **A** y con una `shape` **B**. El proceso sigue estos pasos antes de ejecutar cualquier operación binaria (suma, multiplicación, etc.):

1. **Alineación de dimensiones:** Se comparan las formas de los dos tensores de derecha a izquierda.
2. **Cálculo de la forma resultante (`out_shape`):** Si una dimensión es `1` (o no existe) y la otra es `N`, el tamaño resultante de esa dimensión será `N`.
3. **Creación de las vistas:** Se instancian dos vistas nuevas (una para el tensor A y otra para el tensor B) que compartirán la forma final calculada (`out_shape`).
4. **Asignación de Strides:** Para cada vista, si su dimensión original era de tamaño `1` y fue "expandida" al tamaño `N`, el `stride` correspondiente para esa dimensión en la vista **se fuerza a 0**. Si la dimensión no fue expandida, mantiene su `stride` original.

De esta forma, la función final que ejecuta la suma o la multiplicación punto a punto es completamente agnóstica. Solo recibe dos vistas que, para ella, tienen exactamente la misma forma y número de dimensiones, delegando todo el trabajo complejo de enrutamiento de memoria a los `strides`. Finalmente se adapta cada tensor de la operación con la función `broadcast_to`, muy parecida a la de `reshape`.

## 3. Prueba

Dentro de la carpeta `test/` se encuentra un ejecutable con diferentes pruebas asociadas a este capítulo, para ejecutarlo:

```bash
g++ ../tests/01_test_views.cpp tensor.cpp utils.cpp  -I../include -o test_views && ./test_views
```

En él, pondremos a prueba de forma práctica toda la teoría sobre las vistas y el *broadcasting* que acabamos de ver. Al ejecutarlo, la consola te guiará a través de tres bloques principales:

Primero, verás cómo partimos de un tensor unidimensional básico (un vector de 6 elementos) y, utilizando la operación `reshape`, lo "moldeamos" para que adopte la forma de una matriz de `2x3`. El propio programa realizará una comprobación interna para confirmarte que, efectivamente, esta nueva vista sigue apuntando exactamente al mismo bloque de memoria que el tensor original.

A continuación, observarás cómo esa matriz de dos dimensiones se transforma virtualmente en un tensor de tres dimensiones mediante la función `view_to_3d`. Como comentamos en la teoría, podrás comprobar cómo esta abstracción prepara el tensor para que algoritmos más complejos (como GEMM) puedan procesarlo genéricamente.

Finalmente, presenciarás la mecánica del *broadcasting* en acción. El test tomará un pequeño vector de 3 elementos (`10, 20, 30`) y calculará las formas necesarias para expandirlo a una matriz de `3x3`. Si prestas atención a los detalles de la memoria que imprimirá la función `info` por consola, verás la prueba definitiva de nuestra teoría: **el `stride` de la nueva dimensión expandida es exactamente `0`**, recorriéndose en bucle la misma fila del tensor.