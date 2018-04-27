async function myFirstTfjs() {
  // Create a simple model.
  const model = tf.sequential()
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }))

  // Prepare the model for training: Specify the loss and the optimizer.
  model.compile({
    loss: 'meanSquaredError',
    optimizer: 'sgd'
  })

  const m = document.getElementById('varM').value
  const b = document.getElementById('varB').value
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
  model.fit(xs, ys, { epochs: numEpochs }).then(() => {
    document.getElementById('micro_out_div').innerText = model.predict(
      tf.tensor2d([guessMe], [1, 1])
    )
  })

  // Use the model to do inference on a data point the model hasn't seen.
  // Should print approximately 39.
}
