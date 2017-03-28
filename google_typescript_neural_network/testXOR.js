"use strict";
var nn = require("./nn");
var iter = 0;
function trainAndTest(network, dataPoints, learningRate, regularizationRate) {
    var loss = 0;
    for (var i_1 = 0; i_1 < dataPoints.length; i_1++) {
        var dataPoint = dataPoints[i_1];
        var input = [dataPoint.i1, dataPoint.i2];
        var output = nn.forwardProp(network, input);
        nn.backProp(network, dataPoint.label, nn.Errors.SQUARE);
        nn.updateWeights(network, learningRate, regularizationRate);
        loss += nn.Errors.SQUARE.error(output, dataPoint.label);
    }
    if ((iter++ % 1117) == 0)
        console.log(loss);
    return loss / dataPoints.length;
}
// Make a simple network.
var numInputs = 3;
var shape = [2, 2, 1];
var learningRate = 0.05;
var regularizationRate = 0.0; // turn off since we are not over fitting
var network = nn.buildNetwork(shape, nn.Activations.SIGMOID, nn.Activations.SIGMOID, nn.RegularizationFunction.L1, ["i1", "i2"], false);
var trainingData = [];
trainingData.push({ i1: 0.8, i2: 0.2, label: 1 });
trainingData.push({ i1: 0.2, i2: 0.8, label: 1 });
trainingData.push({ i1: 0.8, i2: 0.8, label: 0 });
trainingData.push({ i1: 0.2, i2: 0.2, label: 0 });
var testingData = [];
testingData.push({ i1: 0.7, i2: 0.15, label: 1 });
testingData.push({ i1: 0.18, i2: 0.85, label: 1 });
testingData.push({ i1: 0.75, i2: 0.81, label: 0 });
testingData.push({ i1: 0.1, i2: 0.2, label: 0 });
var lossTrain = 1;
for (var i = 0; i < 500000 && lossTrain > 0.01; i++)
    lossTrain = trainAndTest(network, trainingData, learningRate, regularizationRate);
console.log("\nTest:\n");
iter = 0; // zero out so console.log printout occurs
var lossTest = trainAndTest(network, testingData, learningRate, regularizationRate);
//# sourceMappingURL=testXOR.js.map