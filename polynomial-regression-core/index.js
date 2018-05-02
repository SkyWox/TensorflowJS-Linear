import * as tf from "@tensorflow/tfjs";
import { generateData } from "./data";
import { plotData, plotDataAndPredictions, renderCoefficients } from "./ui";

/**
 * We want to learn the coefficients that give correct solutions to the
 * following quadratic equation:
 *      y = a * x^3 + b * x^2 + c * x + d
 * In other words we want to learn values for:
 *      a
 *      b
 *      c
 *      d
 * Such that this function produces 'desired outputs' for y when provided
 * with x. We will provide some examples of 'xs' and 'ys' to allow this model
 * to learn what we mean by desired outputs and then use it to produce new
 * values of y that fit the curve implied by our example.
 */

// Step 2. Create an optimizer, we will use this later. You can play
// with some of these values to see how the model perfoms.
const numIterations = 75;
const learningRate = 0.5;
const optimizer = tf.train.sgd(learningRate);

// Step 3. Write our training process functions.

/*
 * This function represents our 'model'. Given an input 'x' it will try and
 * predict the appropriate output 'y'.
 *
 * It is also sometimes referred to as the 'forward' step of our training
 * process. Though we will use the same function for predictions later.
 *
 * @return number predicted y value
 */
function predict(x, coefficients) {
  // y = a * x ^ 3 + b * x ^ 2 + c * x + d
  return tf.tidy(() => {
    return coefficients["3"]
      .mul(x.pow(tf.scalar(3, "int32")))
      .add(coefficients["2"].mul(x.square()))
      .add(coefficients["1"].mul(x))
      .add(coefficients["0"]);
  });
}

/*
 * This will tell us how good the 'prediction' is given what we actually
 * expected.
 *
 * prediction is a tensor with our predicted y values.
 * labels is a tensor with the y values the model should have predicted.
 */
function loss(prediction, labels) {
  // Having a good error function is key for training a machine learning model
  const error = prediction
    .sub(labels)
    .square()
    .mean();
  return error;
}

/*
 * This will iteratively train our model.
 *
 * xs - training data x values
 * ys — training data y values
 */
async function train(xs, ys, coefficients, numIterations) {
  for (let iter = 0; iter < numIterations; iter++) {
    // optimizer.minimize is where the training happens.

    // The function it takes must return a numerical estimate (i.e. loss)
    // of how well we are doing using the current state of
    // the variables we created at the start.

    // This optimizer does the 'backward' step of our training process
    // updating variables defined previously in order to minimize the
    // loss.
    optimizer.minimize(() => {
      // Feed the examples into the model
      const pred = predict(xs, coefficients);
      return loss(pred, ys);
    });

    // Use tf.nextFrame to not block the browser.
    await tf.nextFrame();
  }
}

function getDataSync(coefficients) {
  var sync = {};
  for (var n in coefficients) {
    sync[n] = coefficients[n].dataSync()[0];
  }
  return sync;
}

async function learnCoefficients(trueCoefficients) {
  // These are the things we want the model
  // to learn in order to do prediction accurately. We will initialize
  // them with random values.
  var workingCoefficients = {};
  for (var i = 0; i < 4; i++) {
    workingCoefficients[String(i)] = tf.variable(tf.scalar(Math.random()));
  }

  const trainingData = generateData(100, trueCoefficients);

  // Plot original data
  renderCoefficients("#data .coeff", trueCoefficients);
  await plotData("#data .plot", trainingData.xs, trainingData.ys);

  // See what the predictions look like with random coefficients
  renderCoefficients("#random .coeff", getDataSync(workingCoefficients));
  const predictionsBefore = predict(trainingData.xs, workingCoefficients);
  await plotDataAndPredictions(
    "#random .plot",
    trainingData.xs,
    trainingData.ys,
    predictionsBefore
  );

  // Train the model!
  await train(
    trainingData.xs,
    trainingData.ys,
    workingCoefficients,
    numIterations
  );

  // See what the final results predictions are after training.
  renderCoefficients("#trained .coeff", getDataSync(workingCoefficients));
  const predictionsAfter = predict(trainingData.xs, workingCoefficients);
  await plotDataAndPredictions(
    "#trained .plot",
    trainingData.xs,
    trainingData.ys,
    predictionsAfter
  );

  predictionsBefore.dispose();
  predictionsAfter.dispose();
}

learnCoefficients({ 3: -0.8, 2: -0.2, 1: 0.9, 0: 0.5 });
