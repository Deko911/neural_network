import * as wasm from 'neural_network'

let data = new Float32Array([0, 0, 0, 255, 255, 255, 0, 0, 255, 255, 0, 0, 0, 255, 0],);
let targets = new Float32Array([1, 0, 1, 1, 0]);

let model = new wasm.PerceptronJS(3, 0.2)
model.fit(data, targets, 15);

console.log(model.predict(new Float32Array([17, 17, 17])));

let picker = document.getElementById("picker")! as HTMLInputElement;
let text = document.getElementById("font")!;
let background = document.getElementById("color")!;

function predictFont() {
    let color = picker.value.replace("#", "");
    background.style.backgroundColor = picker.value;
    let r = parseInt(color[0] + color[1], 16)
    let g = parseInt(color[2] + color[3], 16)
    let b = parseInt(color[4] + color[5], 16)

    let font = model.predict(new Float32Array([r, g, b]))[0]

    if (font > 0.5) {
        text.style.color = "#fff"
    }else {
        text.style.color = "#000"
    }
}

picker.onchange = predictFont