# TensorflowJS-Linear

<!-- TOC -->

* [TensorflowJS-Linear](#tensorflowjs-linear)
  * [What is it?](#what-is-it)
  * [[Demo](http://skywox.me/TensorflowJS-Linear/)](#demohttpskywoxmetensorflowjs-linear)
  * [How to use:](#how-to-use)
  * [What's happening?](#whats-happening)
  * [Running locally](#running-locally)
  * [Improvements](#improvements)

<!-- /TOC -->

## What is it?

An implementation of Tensorflow JS that will try to guess a correct Y value for a linear equation you build.

## [Demo](http://skywox.me/TensorflowJS-Linear/)

## How to use:

Build a linear equation in the form y = ax + b.  
Other parameters:  
|Parameter|Description|
|---|---|
|Sample points | How many (x,y) pairs are generated from your equation and fed to the algorithm|
|Epochs | How many "generations" of the algorithm will be trained on the data before answering|
|X to predict | The x value you want the algorithm to predict the Y value for|

## What's happening?

Machine learning in your browser! This uses Tensorflow JS to implement a simple neural network.

## Running locally

Assuming you use yarn:

```sh
yarn
yarn watch
```

Default serving location is `localhost:1234`

## Improvements

* Better UI
* Graphs of data or Tensorboard
* Support higher order functions
