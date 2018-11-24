/*
 * Copyright 2018 Google LLC.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 */
package org.tensorflow.contrib.examples.basic;

import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Shape;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Add;
import org.tensorflow.op.core.Placeholder;

final class Main {

  public static void main(String[] args) throws Exception {
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
    }
  }

  private Main() {}
}
