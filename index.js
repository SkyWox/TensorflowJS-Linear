import * as tf from '@tensorflow/tfjs'
import {generateData} from './data'
import {plotData, plotDataAndPredictions, renderCoefficients} from './ui'

function getPolyEq() {
  const polyEq = document
    .getElementById('polyEq')
    .value.replace(/ /g, '')
    .split(/(?=[$-/:-?{-~!"_`\[\]])/gi)

  //mimic defaultDict in python
  var coeffs = new Proxy(
    {},
    {
      get: (target, name) => (name in target ? target[name] : 0)
    }
  )

  polyEq.forEach(phrase => {
    const [coeff, base, exp] = phrase.split(/(?=[x^])/gi)
    if (base) {
      if (exp) {
        coeffs[Number(exp.substring(1))] += Number(coeff)
      } else {
        coeffs[1] += Number(coeff)
      }
    } else {
      if (phrase[0] === '+') {
        coeffs[0] += Number(phrase.substring(1))
      } else if (phrase[0] === '-') {
        coeffs[0] -= Number(phrase.substring(1))
      } else {
        coeffs[0] += Number(phrase)
      }
    }
  })

  // Make sure all slots are filled b/c js dicts hide their 0s
  for (let i = 0; i <= maxOrder(coeffs); i++) {
    if (coeffs[i] == 0) {
      coeffs[i] = 0
    }
  }

  return coeffs
}

const learningRate = 0.5
const optimizer = tf.train.sgd(learningRate)

function predict(x, coefficients) {
  // y = a * x ^ 3 + b * x ^ 2 + c * x + d
  return tf.tidy(() => {
    return coefficients[3]
      .mul(x.pow(tf.scalar(3, 'int32')))
      .add(coefficients[2].mul(x.square()))
      .add(coefficients[1].mul(x))
      .add(coefficients[0])
  })
}

function loss(prediction, labels) {
  // Having a good error function is key for training a machine learning model
  const error = prediction
    .sub(labels)
    .square()
    .mean()
  return error
}

async function train(xs, ys, coefficients, numEpochs) {
  for (let iter = 0; iter < numEpochs; iter++) {
    // optimizer.minimize is where the training happens.

    // The function it takes must return a numerical estimate (i.e. loss)
    // of how well we are doing using the current state of
    // the variables we created at the start.

    // This optimizer does the 'backward' step of our training process
    // updating variables defined previously in order to minimize the
    // loss.
    optimizer.minimize(() => {
      // Feed the examples into the model
      const pred = predict(xs, coefficients)
      return loss(pred, ys)
    })

    // Use tf.nextFrame to not block the browser.
    await tf.nextFrame()
  }
}

function getDataSync(coefficients) {
  var sync = {}
  for (let n in coefficients) {
    sync[n] = coefficients[n].dataSync()[0]
  }
  return sync
}

function coefficientSubset(coefficients, highest) {
  var coeff = coefficients
  for (let i = highest + 1; i <= maxOrder(coefficients); i++) {
    coeff[i] = tf.variable(tf.scalar(0))
  }
  return coeff
}

function maxOrder(coeff) {
  // Hardwired to return no less than 3 so that all neccessary tensors get built to run the model
  return Math.max(
    3,
    Object.keys(coeff).reduce(function(a, b) {
      return a > b ? a : b
    })
  )
}

function randomCoefficients(order, maximumOrder) {
  var coefficients = {}
  for (let i = 0; i <= order; i++) {
    coefficients[String(i)] = tf.variable(tf.scalar(Math.random()))
  }
  for (let i = order + 1; i <= maximumOrder; i++) {
    const buffer = tf.buffer([1])
    buffer.set(0, 0)
    coefficients[String(i)] = buffer.toTensor()
  }
  return coefficients
}

async function myFirstTfjs() {
  const trueCoefficients = getPolyEq()
  console.log(trueCoefficients)

  const numData = Number(document.getElementById('numData').value)
  const numEpochs = document.getElementById('numEpochs').value
  // These are the things we want the model
  // to learn in order to do prediction accurately. We will initialize
  // them with random values.
  const maximumOrder = maxOrder(trueCoefficients)
  const trainingData = generateData(numData, trueCoefficients)

  // Plot original data
  renderCoefficients('#data .coeff', trueCoefficients)
  await plotData('#data .plot', trainingData.xs, trainingData.ys)

  for (let i = 1; i <= maximumOrder; i++) {
    const workingCoefficients = randomCoefficients(i, maximumOrder)
    train(
      trainingData.xs,
      trainingData.ys,
      workingCoefficients,
      numEpochs
    ).then(async () => {
      renderCoefficients(
        '#out' + i + ' .coeff',
        getDataSync(workingCoefficients)
      )
      const predictionsBefore = predict(trainingData.xs, workingCoefficients)
      await plotDataAndPredictions(
        '#out' + i + ' .plot',
        trainingData.xs,
        trainingData.ys,
        predictionsBefore
      )
      predictionsBefore.dispose()
    })
  }
}

document.getElementById('guessBtn').addEventListener('click', myFirstTfjs)
