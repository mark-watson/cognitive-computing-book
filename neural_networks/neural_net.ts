/**
 * Copyright 2016 Mark Watson. All rights reserved.
 * This code may be used under the Apache 2 license.
 * This notice must remain in this file and derived files.
 */

class BackProp {
  learningRate_w1 = 0.025;
  learningRate_w2 = 0.0125;
  allowedError = 0.05;
  inputs = [];  // numInput
  hidden = [];  // numHidden
  outputs = []; // numOutput
  w1 = [];      // numInput  * numHidden
  w2 = [];      // numHidden * numOutput
  output_errors = []; // numOutput
  hidden_errors = []; // numHidden),
  input_training_examples = [];
  output_training_examples = [];

  constructor(public numInput: number,
              public numHidden: number,
              public numOutput: number) {
    for (var i = 0; i < numInput; i += 1) {
      this.w1[i] = []; // numHidden
      this.inputs[i] = 0;
    }
    for (var i = 0; i < numHidden; i += 1) {
      this.w2[i] = []; // numOutput);
    }
    for (var i = 0; i < numInput; i += 1) {
      for (h = 0; h < numHidden; h += 1) {
        this.w1[i][h] = 0.1 * (Math.random() - 0.05);
      }
    }
    for (var h = 0; h < numHidden; h += 1) {
      for (var o = 0; o < numOutput; o += 1) {
        this.w2[h][o] = 0.05 * (Math.random() - 0.025);
      }
    }
  }    

  add_training_example(inputs, outputs) {
    this.input_training_examples.push(inputs);
    this.output_training_examples.push(outputs);
  }

  sigmoid(x) {
    return (1.0 / (1.0 + Math.exp(-x)));
  }

  sigmoidP(x) {
    var z = this.sigmoid(x);
    return (z * (1.0 - z));
  }

  forward_pass() {
    for (var h = 0; h < this.numHidden; h += 1) {
      this.hidden[h] = 0;
    }
    for (var i = 0; i < this.numInput; i += 1) {
      for (var h = 0; h < this.numHidden; h += 1) {
        this.hidden[h] += this.inputs[i] * this.w1[i][h];
      }
    }
    for (var o = 0; o < this.numOutput; o += 1) {
      this.outputs[o] = 0;
    }
    for (var h = 0; h < this.numHidden; h += 1) {
      for (var o = 0; o < this.numOutput; o += 1) {
        this.outputs[o] += this.sigmoid(this.hidden[h]) * this.w2[h][o];
      }
    }
    for (var o = 0; o < this.numOutput; o += 1) {
        this.outputs[o] = this.sigmoid(this.outputs[o]);
    }
    //if (isNaN(this.outputs[0])) {
    //  console.log("NaN");
    //}
  }

  reset_weights() {
    console.log("* this.numInput=" + this.numInput + ", this.numHidden=" + this.numHidden + ", this.numOutput=" + this.numOutput);
    //console.log(this);
    for (var i = 0; i < this.numInput; i += 1) {
      for (var h = 0; h < this.numHidden; h += 1) {
        this.w1[i][h] = 0.025 * (Math.random() - 0.5);
      }
    }
    for (var h = 0; h < this.numHidden; h += 1) {
      for (var o = 0; o < this.numOutput; o += 1) {
        this.w2[h][o] = 0.005 * (Math.random() - 0.5);
      }
    }
  }

  train_helper() {
    var error = 0, outs;
    var num_cases = this.input_training_examples.length;
    for (var ncase = 0; ncase < num_cases; ncase += 1) {
      // zero out the errors: at the hidden and output layers:
      for (var h = 0; h < this.numHidden; h += 1) {
        this.hidden_errors[h] = 0;
      }
      for (var o = 0; o < this.numOutput; o += 1) {
        this.output_errors[o] = 0;
      }
      for (var i = 0; i < this.numInput; i += 1) {
        this.inputs[i] = this.input_training_examples[ncase][i];
      }
      outs = this.output_training_examples[ncase];
      this.forward_pass();
      for (var o = 0; o < this.numOutput; o += 1) {
        this.output_errors[o] = (outs[o] - this.outputs[o]) * this.sigmoidP(this.outputs[o]);
      }
      for (var h = 0; h < this.numHidden; h += 1) {
        this.hidden_errors[h] = 0.0;
        for (var o = 0; o < this.numOutput; o += 1) {
          this.hidden_errors[h] += this.output_errors[o] * this.w2[h][o];
        }
      }
      for (var h = 0; h < this.numHidden; h += 1) {
        this.hidden_errors[h] =
          this.hidden_errors[h] * this.sigmoidP(this.hidden[h]);
      }
      // update the hidden to output weights:
      for (var o = 0; o < this.numOutput; o += 1) {
        for (var h = 0; h < this.numHidden; h += 1) {
          this.w2[h][o] +=
          this.learningRate_w2 * this.output_errors[o] * this.hidden[h];
        }
      }
      // update the input to hidden weights:
      for (var h = 0; h < this.numHidden; h += 1) {
        for (var i = 0; i < this.numInput; i += 1) {
          this.w1[i][h] +=
          this.learningRate_w1 * this.hidden_errors[h] * this.inputs[i];
        }
      }
      for (var o = 0; o < this.numOutput; o += 1) {
        error += Math.abs(outs[o] - this.outputs[o]);
      }
    }
    return error;
  }

  recall(inputs) {
    var i, numInputs = this.inputs.length;
    for (i = 0; i < numInputs; i += 1) {
      this.inputs[i] = inputs[i];
    }
    this.forward_pass();
    return this.outputs;
  }

  train() {
    var error;
    for (var iter = 1; iter < 500000; iter += 1) {
      error = this.train_helper();
      if ((iter % 1800) === 0) {
          console.log(error);
      }
      if (error < this.allowedError) {
          break;
      }
      // reset weights: if network get stuck then start over:
      if (error > 2.0  && (iter % 17111) === 0) {
          console.log("** reset weights");
          this.reset_weights();
      }
    }
  }

}

function test_nn() {
  var test_network = new BackProp(3, 3, 3);
  test_network.add_training_example([0.1, 0.1, 0.9], [0.9, 0.1, 0.1]);
  test_network.add_training_example([0.1, 0.9, 0.1], [0.1, 0.1, 0.9]);
  test_network.add_training_example([0.9, 0.1, 0.1], [0.1, 0.9, 0.1]);
  test_network.train();
  console.log(test_network.recall([0.08, 0.2, 0.88]));
  console.log(test_network.recall([0.93, 0.2, 0.11]));
  console.log(test_network.recall([0.11, 0.9, 0.06]));
}

// simple test:
test_nn();

