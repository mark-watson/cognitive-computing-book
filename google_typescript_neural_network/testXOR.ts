import * as nn from "./nn";

type Data = {
  i1: number,
  i2: number,
  label: number
};

var iter = 0;

function trainAndTest(network: nn.Node[][],
                      dataPoints: Data[],
                      learningRate: number,
                      regularizationRate: number): number {
  let loss = 0;
  for (let i = 0; i < dataPoints.length; i++) {
    let dataPoint = dataPoints[i];
    let input = [dataPoint.i1, dataPoint.i2];
    let output = nn.forwardProp(network, input);
    nn.backProp(network, dataPoint.label, nn.Errors.SQUARE);
    nn.updateWeights(network, learningRate, regularizationRate);
    loss += nn.Errors.SQUARE.error(output, dataPoint.label);
  }
  if ((iter++ % 1117) == 0)  console.log(loss);
  return loss / dataPoints.length;
}

// Make a simple network
let numInputs = 3;
let shape = [2,2,1];
let learningRate = 0.15;
let regularizationRate = 0.0; // turn off since we are not over fitting

var network = nn.buildNetwork(shape, nn.Activations.SIGMOID,
                              nn.Activations.SIGMOID,
                              nn.RegularizationFunction.L1,
                              ["i1", "i2"], false);
var trainingData : Data[] = [];
trainingData.push({i1: 0.8, i2: 0.2, label: 1});
trainingData.push({i1: 0.2, i2: 0.8, label: 1});
trainingData.push({i1: 0.8, i2: 0.8, label: 0});
trainingData.push({i1: 0.2, i2: 0.2, label: 0});

var testingData : Data[] = [];
testingData.push({i1: 0.7, i2: 0.15, label: 1});
testingData.push({i1: 0.18, i2: 0.85, label: 1});
testingData.push({i1: 0.75, i2: 0.81, label: 0});
testingData.push({i1: 0.1, i2: 0.2, label: 0});

var lossTrain : Number = 1;
for (var i=0; i<80000 && lossTrain > 0.001; i++)
  lossTrain = trainAndTest(network, trainingData,
                           learningRate, regularizationRate);
console.log("\nTest:\n");
iter = 0; // zero out so console.log printout occurs
let lossTest = trainAndTest(network, testingData,
                            learningRate, regularizationRate);

