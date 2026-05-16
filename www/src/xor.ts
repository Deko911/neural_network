import * as wasm from 'neural_network'

let data = new Float32Array([0, 0, 1, 0,0, 1, 1, 1])
let targets = new Float32Array([0, 1, 1, 0])

let model = new wasm.NeuralNetworkJS(2, 5, new Uint32Array([2, 8, 4, 1]), [wasm.ACTIVATIONS.RELU, wasm.ACTIVATIONS.RELU, wasm.ACTIVATIONS.SIGMOID], [wasm.INITIALIZER.HE, wasm.INITIALIZER.HE, wasm.INITIALIZER.XAVIER], wasm.LOSS.BINARY_CROSS_ENTROPY)

function colorResults () {
    for (let i = 0; i < 2; i++) {
        for (let j = 0; j < 2; j++) {
            let result = model.predict(new Float32Array([i, j]))[0]
            document.getElementById(`${i}x${j}`)!.style.backgroundColor = `rgb(${150 + 100 * (1 - result)}, ${150 + 100 * result}, 150)`
        }
    }
}

function trainSteps () {
    model.fit(data, targets, 1, 0)
    colorResults()
}

document.getElementById('trainBtn')!.onclick = trainSteps
colorResults()