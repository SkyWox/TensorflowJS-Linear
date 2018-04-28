import * as tf from '@tensorflow/tfjs'

export function generateData(numPoints, coeff, sigma = 0.04) {
  return tf.tidy(() => {
    var coe = {}
    for (var n in coeff) {
      coe[n] = tf.scalar(coeff[n])
    }

    const xs = tf.randomUniform([numPoints], -1, 1)

    //
    // So looping over the .add function doesn't add in the terms.
    // I think it's because it doesn't AWAIT for the for loop to finish
    //

    // Generate polynomial data
    var ys = tf
      .randomNormal([numPoints], 0, sigma) // Add random noise to the generated data
      /*
    for (var c in coe) {
      console.log(Number(c))
      console.log(coe[c])
      ys.add(coe[c].mul(xs.pow(tf.scalar(Number(c), 'int32'))))
    }
    */
      .add(coe['0'].mul(xs.pow(tf.scalar(0, 'int32'))))
      .add(coe['3'].mul(xs.pow(tf.scalar(3, 'int32'))))
      .add(coe['2'].mul(xs.pow(tf.scalar(2, 'int32'))))
      .add(coe['1'].mul(xs.pow(tf.scalar(1, 'int32'))))

    // Normalize the y values to the range 0 to 1.
    const ymin = ys.min()
    const ymax = ys.max()
    const yrange = ymax.sub(ymin)
    const ysNormalized = ys.sub(ymin).div(yrange)

    return {
      xs,
      ys: ysNormalized
    }
  })
}
