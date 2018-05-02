import * as tf from '@tensorflow/tfjs'

export function generateData(numPoints, coeff, sigma = 0.04) {
  return tf.tidy(() => {
    var coe = {}
    for (var n in coeff) {
      coe[n] = tf.scalar(coeff[n])
    }

    const xs = tf.randomUniform([numPoints], -1, 1)

    // Generate polynomial data - can't add sequentially - has to be one go
    var ys = tf
      .randomNormal([numPoints], 0, sigma) // Add random noise to the generated data
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
