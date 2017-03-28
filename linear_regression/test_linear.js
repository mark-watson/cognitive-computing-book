var regression = require('./regression.js');

var data = [

	[4,4.2],
	[5,6],
	[6,8],
	[7,5],
  [8,7],
  [9,9],
  [10,8],
  [11,8.2],
  [12,11],
  [13,7],
  [14,10]
];

// fit a straight line through the data:
var reg1 = regression('polynomial', data, 1);

console.log("\mreg1: " + reg1.string);
