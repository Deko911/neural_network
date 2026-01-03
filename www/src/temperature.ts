import * as wasm from 'neural_network'

let data = new Float32Array([-20, -10, 0, 10, 20]);
let targets = new Float32Array([-4, 14, 32, 50 ,68]);

let model = new wasm.PerceptronJS(1, 1.0)
model.fit(data, targets, 15);


let temperatureRange = document.getElementById("temperature")! as HTMLInputElement;

let previous = temperatureRange.value;

function predictTemperature() {
    let celsius = temperatureRange.value;
    if (previous == celsius) {
        return;
    }
    previous = celsius;

    let farenheit = model.predict(new Float32Array([Number(celsius)]));
    document.getElementById("celsius")!.innerText = "celsius:" + celsius + "°";
    document.getElementById("farenheit")!.innerText = "farenheit: " + String(Math.trunc(farenheit * 100) / 100) + "°";
}

temperatureRange.onmousemove = predictTemperature