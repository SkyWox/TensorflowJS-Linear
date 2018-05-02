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
        coeffs[exp.substring(1)] += Number(coeff)
      } else {
        coeffs['0'] += Number(coeff)
      }
    } else {
      if (phrase[0] === '+') {
        coeffs.const += Number(phrase.substring(1))
      } else if (phrase[0] === '-') {
        coeffs.const -= Number(phrase.substring(1))
      } else {
        coeffs.const += Number(phrase)
      }
    }
  })

  return coeffs
}

async function myFirstTfjs() {
  // Plot data
  const trueCoefficients = {3: -0.8, 2: -0.2, 1: 0.9, 0: 0.5}
  const trainingData = generateData(100, trueCoefficients)

  // Plot original data
  renderCoefficients('#data .coeff', trueCoefficients)
  await plotData('#data .plot', trainingData.xs, trainingData.ys)

  // Create a simple model.
  const model = tf.sequential()
  model.add(tf.layers.dense({units: 1, inputShape: [1]}))

  // Prepare the model for training: Specify the loss and the optimizer.
  model.compile({
    loss: 'meanSquaredError',
    optimizer: 'sgd'
  })

  const coeffs = getPolyEq()

  const m = coeffs['0']
  const b = coeffs.const
  const numData = document.getElementById('numData').value
  const numEpochs = document.getElementById('numEpochs').value
  const guessMe = document.getElementById('guessMe').value

  document.getElementById('correct_answer').innerHTML += m * guessMe + Number(b)
  // Generate some synthetic data for training. (y = 2x - 1)
  var x = []
  const range = [0, 10]
  const step = Math.floor((range[1] - range[0]) / numData)

  for (let i = 0; i < numData; i++) {
    x.push(range[0] + step * i)
  }
  var y = []
  x.map(xx => {
    y.push(m * xx + Number(b))
  })

  const xs = tf.tensor2d(x, [x.length, 1])
  const ys = tf.tensor2d(y, [y.length, 1])

  // Train the model using the data.
  model.fit(xs, ys, {epochs: numEpochs}).then(() => {
    document.getElementById('micro_out_div').innerText = model.predict(
      tf.tensor2d([guessMe], [1, 1])
    )
  })
}

document.getElementById('guessBtn').addEventListener('click', myFirstTfjs)
