# TensorFlow basics

This is a low-level introduction to TensorFlow through its Java
bindings.

TensorFlow at its core is just a high-performance, scalable numerical
library.

Although it is typically used for machine learning, here we focus on
its basic number-crunching abstractions.

The first thing to note is that TensorFlow is a
[dataflow](https://en.wikipedia.org/wiki/Dataflow_programming) system.

The way TensorFlow views the world, computation is organized as a
[graph](https://www.tensorflow.org/guide/graphs), with primitive
operators as nodes in this graph. For example, operators like _add_
and _multiply_ could be nodes in this graph.

Data _flows_ through this graph. TensorFlow can parallelize and
pipeline computation as you _feed_ large amounts of data at one end,
and _fetch_ the results at the other end, so it's pretty awesome at
processing huge amounts of data and distributing the calculations.

On the flip side, when you use TensorFlow you don't really write code
in the usual way. You first construct a graph of operators
representing your computation. Then you _feed_ the inputs of your
graph with your data and _fetch_ results at its output nodes.

By the way, what exactly is _data_?

TensorFlow has just one idea of data: a _multi-dimensional typed
array_. For example, `int[3]` or `float[256][256]` are types of data
TensorFlow uses. Note that all elements of this array must have the
same type. TensorFlow doesn't support having array elements with
different types.

All operators generally take in such arrays as parameters, and produce
other arrays as outputs.

TensorFlow calls these arrays _tensors_ (and of course, because it
flows through the graph we get TensorFlow.) But for all practical
purposes you can view them as multi-dimensional arrays.

There are a few Java objects and methods in Tensorflow to refer to such arrays.

A
[Tensor](https://www.tensorflow.org/api_docs/java/reference/org/tensorflow/Tensor)
is a wrapper for multi-dimensional arrays. For instance, it could wrap
an `int` array like `int[2][5][7]`.

The
[DataType](https://www.tensorflow.org/api_docs/java/reference/org/tensorflow/Tensor.html#dataType())
for a Tensor is just the underlying type for the array. For instance,
the Tensor wrapping `int[2][5][7]` would have `DataType.INT32`.

The
[rank](https://www.tensorflow.org/api_docs/java/reference/org/tensorflow/Tensor.html#numDimensions())
of a Tensor is the number of dimensions in its underlying array. For
instance, `int[2][5][7]` has the rank 3. TensorFlow also supports
scalars, and the rank of a scalar is 0.

The
[Shape](https://www.tensorflow.org/api_docs/java/reference/org/tensorflow/Shape)
is an integer array containing the size of each dimension. For
instance, the shape of `int[2][5][7]` is `[2, 5, 7]`.

---

*Pop quiz* What is rank(shape(any tensor))?

---

To recap.

A "TensorFlow program" has three core abstractions:

- a
   [Graph](https://www.tensorflow.org/api_docs/java/reference/org/tensorflow/Graph),
   which has nodes called
- [Operations](https://www.tensorflow.org/api_docs/java/reference/org/tensorflow/Operation)
   that connect to other Operations through edges that pass
- [Tensors](https://www.tensorflow.org/api_docs/java/reference/org/tensorflow/Tensor),
   or typed multi-dim arrays

Let's put this together into a simple TensorFlow program that adds two arrays.

```java
    try (Graph g = new Graph()) {
      // A convenience class to build nodes in the graph.
      Ops ops = Ops.create(g);

      // Placeholders are nodes where you can feed inputs to the graph. Here we create
      // two such nodes, of type int[3].
      Placeholder<Integer> a = ops.placeholder(Integer.class, Placeholder.shape(Shape.make(3)));
      Placeholder<Integer> b = ops.placeholder(Integer.class, Placeholder.shape(Shape.make(3)));
      Add<Integer> add = ops.add(a, b);
      // Our graph now looks like
      //   a        b
      //    \      /
      //     \    /
      //       add
```

At this point, we've just constructed a graph. Now we need to evaluate
it, by feeding in some input values for `a` and `b`, and fetching the
output from `add`.

```java
      // A session is a handle to APIs that let you feed and fetch
      // inputs and outputs from the graph.
      try (Session s = new Session(g)) {
        // Create two int[3] tensors for the input
        Tensor<Integer> ain = Tensors.create(new int[] {1, 2, 3});
        Tensor<Integer> bin = Tensors.create(new int[] {4, 5, 6});

        Session.Runner r = s.runner().feed(a.asOutput(), ain).feed(b.asOutput(), bin).fetch(add);
        Tensor<Integer> out = r.run().get(0).expect(Integer.class);
        // Create an int[] array to capture the output.
        int[] result = new int[3];
        out.copyTo(result);
        for (int i = 0; i < result.length; i++) {
          System.out.println(String.format("[%d] = %d", i, result[i]));
        }
      }
```

The full code is in
[`Main.java`](src/main/java/org/tensorflow/contrib/learn/examples/basic/Main.java). You
can test it by running

```
$ ./gradlew run
```
in this directory.

For me, the task outputs
```
$ ./gradlew run
> Task :run
2018-11-23 18:30:46.858287: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
[0] = 5
[1] = 7
[2] = 9
```
You may ignore the informational message about the TensorFlow binary.
It just notes that the default library pulled from Maven (a CPU-based
implementation) doesn't make full use of the available instruction
set.
